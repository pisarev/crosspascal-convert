{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35GGUF                                                     }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35GGUF;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.GGUF, CrossPascal.Qwen35Source;

type
  TQwen35GGUFBinding = record
    Layer: Integer;
    Name: string;
    SourceName: string;
    Shape: TArray<Int64>;
    Tensor: TGGUFTensor;
  end;

function ReadQwen35GGUFConfig(const Source: TGGUFCheckpoint): TQwen35Config;
function ReadQwen35GGUFBindings(const Source: TGGUFCheckpoint;
  const Config: TQwen35Config): TArray<TQwen35GGUFBinding>;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Generics.Collections{$ELSE}System.SysUtils, System.Math, System.Generics.Collections{$ENDIF},
  TVM.CpuGraph;

function ReadQwen35GGUFConfig(const Source: TGGUFCheckpoint): TQwen35Config;
const
  Prefix = 'qwen35.';
var
  Blocks, Interval, Inner, I, Sum: Integer;
  Sections: TArray<Int32>;

  function Dimension(const Key: string; const Minimum, Maximum: Integer): Integer;
  var
    Value: UInt32;
  begin
    Value := Source.Unsigned(Prefix + Key);
    RequireCpu((Value >= UInt32(Minimum)) and (Value <= UInt32(Maximum)),
      'Qwen GGUF dimension is out of range: ' + Key);
    Result := Integer(Value);
  end;

begin
  RequireCpu(Source <> nil, 'Qwen GGUF source is required');
  RequireCpu(Source.Text('general.architecture') = 'qwen35', 'Expected dense Qwen 3.5/3.8 GGUF');
  Result := Default(TQwen35Config);
  Blocks := Dimension('block_count', 1, 2048);
  Result.MtpLayers := Source.Unsigned(Prefix + 'nextn_predict_layers', 0);
  RequireCpu((Result.MtpLayers >= 0) and (Result.MtpLayers < Blocks), 'Invalid Qwen GGUF MTP layer count');
  Result.Layers := Blocks - Result.MtpLayers;
  RequireCpu(Result.Layers <= 1024, 'Qwen GGUF decoder has too many layers');
  Result.Hidden := Dimension('embedding_length', 1, 65536);
  Result.Intermediate := Dimension('feed_forward_length', 1, 1048576);
  Result.Context := Dimension('context_length', 1, High(Integer));
  Result.Heads := Dimension('attention.head_count', 1, 1024);
  Result.KVHeads := Dimension('attention.head_count_kv', 1, Result.Heads);
  Result.HeadSize := Dimension('attention.key_length', 1, 65536);
  RequireCpu(Dimension('attention.value_length', 1, 65536) = Result.HeadSize,
    'Qwen GGUF full attention key and value widths differ');
  Result.RotaryDimensions := Dimension('rope.dimension_count', 2, Result.HeadSize);
  Result.LinearHeads := Dimension('ssm.time_step_rank', 1, 1024);
  Result.LinearKVHeads := Dimension('ssm.group_count', 1, Result.LinearHeads);
  Result.LinearKeySize := Dimension('ssm.state_size', 1, 65536);
  Inner := Dimension('ssm.inner_size', 1, 1048576);
  RequireCpu(Inner mod Result.LinearHeads = 0, 'Qwen GGUF value heads do not divide inner size');
  Result.LinearValueSize := Inner div Result.LinearHeads;
  Result.ConvKernel := Dimension('ssm.conv_kernel', 1, 1024);
  RequireCpu((Result.Heads mod Result.KVHeads = 0) and
    (Result.LinearHeads mod Result.LinearKVHeads = 0) and
    (Result.RotaryDimensions mod 2 = 0), 'Invalid Qwen GGUF head or rotary groups');
  Result.RMSEpsilon := Source.Float(Prefix + 'attention.layer_norm_rms_epsilon');
  Result.RopeFrequency := Source.Float(Prefix + 'rope.freq_base');
  RequireCpu((Result.RMSEpsilon > 0) and not IsNan(Result.RMSEpsilon) and
    not IsInfinite(Result.RMSEpsilon) and (Result.RopeFrequency > 0) and
    not IsNan(Result.RopeFrequency) and not IsInfinite(Result.RopeFrequency),
    'Invalid Qwen GGUF normalization or RoPE value');
  Sections := Source.IntArray(Prefix + 'rope.dimension_sections');
  RequireCpu((Length(Sections) = 4) and (Sections[3] = 0), 'Unsupported Qwen GGUF MRoPE sections');
  Sum := 0;
  for I := 0 to High(Sections) do
  begin
    RequireCpu((Sections[I] >= 0) and (Sections[I] <= Result.HeadSize), 'Invalid Qwen GGUF MRoPE section');
    Inc(Sum, Sections[I]);
  end;
  RequireCpu(2 * Sum = Result.RotaryDimensions, 'Qwen GGUF MRoPE sections do not cover rotary width');
  for I := 0 to 2 do Result.MRopeSections[I] := Sections[I];
  Result.Vocab := Length(Source.TokenPieces('tokenizer.ggml.tokens'));
  RequireCpu((Result.Vocab >= 256) and (Result.Vocab <= 16777216), 'Invalid Qwen GGUF vocabulary size');
  Interval := Dimension('full_attention_interval', 1, Result.Layers);
  SetLength(Result.Attention, Result.Layers);
  for I := 0 to Result.Layers - 1 do
    if (I + 1) mod Interval = 0 then
      Result.Attention[I] := qaFullAttention
    else
      Result.Attention[I] := qaGatedDeltaNet;
  Result.OutputGateType := 'sigmoid';
end;

function ReadQwen35GGUFBindings(const Source: TGGUFCheckpoint;
  const Config: TQwen35Config): TArray<TQwen35GGUFBinding>;
var
  Bindings: TArray<TQwen35GGUFBinding>;
  Used: TDictionary<string, Boolean>;
  Layer, Count, I: Integer;
  Prefix, LayerText: string;
  QWidth, KVWidth, LinearQK, LinearV, ConvWidth: Int64;
  Info: TGGUFTensor;

  procedure Add(const Name, SourceName: string; const Shape: array of Int64;
    const SqueezedConv: Boolean = False);
  var
    Tensor: TGGUFTensor;
    J, Rank: Integer;
    NativeShape: TArray<Int64>;
  begin
    Tensor := Source.Tensor(SourceName);
    SetLength(NativeShape, Length(Shape));
    for J := 0 to High(Shape) do NativeShape[J] := Shape[J];
    if SqueezedConv then NativeShape := [Shape[0], Shape[2]];
    Rank := Length(NativeShape);
    RequireCpu((Rank >= 1) and (Rank <= 4) and (Tensor.Elements = TensorElements(Shape)),
      'Qwen GGUF tensor element count differs: ' + SourceName);
    for J := 0 to Rank - 1 do
      RequireCpu(Tensor.Dimensions[J] = NativeShape[Rank - 1 - J],
        'Qwen GGUF tensor shape differs: ' + SourceName);
    for J := Rank to 3 do RequireCpu(Tensor.Dimensions[J] = 1, 'Unexpected Qwen GGUF tensor rank');
    Bindings[Count].Layer := Layer;
    Bindings[Count].Name := Name;
    Bindings[Count].SourceName := SourceName;
    Bindings[Count].Tensor := Tensor;
    SetLength(Bindings[Count].Shape, Length(Shape));
    for J := 0 to High(Shape) do Bindings[Count].Shape[J] := Shape[J];
    Used.Add(SourceName, True);
    Inc(Count);
  end;

begin
  RequireCpu((Source <> nil) and (Config.Layers > 0) and
    (Length(Config.Attention) = Config.Layers), 'Invalid Qwen GGUF source contract');
  SetLength(Bindings, 3 + Config.Layers * 14);
  Used := TDictionary<string, Boolean>.Create;
  try
    Count := 0;
    Layer := -1;
    Add('embed_tokens.weight', 'token_embd.weight', [Config.Vocab, Config.Hidden]);
    Add('norm.weight', 'output_norm.weight', [Config.Hidden]);
    Add('lm_head.weight', 'output.weight', [Config.Vocab, Config.Hidden]);
    QWidth := Int64(Config.Heads) * Config.HeadSize;
    KVWidth := Int64(Config.KVHeads) * Config.HeadSize;
    LinearQK := Int64(Config.LinearKVHeads) * Config.LinearKeySize;
    LinearV := Int64(Config.LinearHeads) * Config.LinearValueSize;
    ConvWidth := 2 * LinearQK + LinearV;
    for Layer := 0 to Config.Layers - 1 do
    begin
      Prefix := 'blk.' + IntToStr(Layer) + '.';
      Add('input_layernorm.weight', Prefix + 'attn_norm.weight', [Config.Hidden]);
      Add('post_attention_layernorm.weight', Prefix + 'post_attention_norm.weight', [Config.Hidden]);
      Add('mlp.gate_proj.weight', Prefix + 'ffn_gate.weight', [Config.Intermediate, Config.Hidden]);
      Add('mlp.up_proj.weight', Prefix + 'ffn_up.weight', [Config.Intermediate, Config.Hidden]);
      Add('mlp.down_proj.weight', Prefix + 'ffn_down.weight', [Config.Hidden, Config.Intermediate]);
      if Config.Attention[Layer] = qaGatedDeltaNet then
      begin
        Add('linear_attn.in_proj_qkv.weight', Prefix + 'attn_qkv.weight', [ConvWidth, Config.Hidden]);
        Add('linear_attn.in_proj_z.weight', Prefix + 'attn_gate.weight', [LinearV, Config.Hidden]);
        Add('linear_attn.in_proj_a.weight', Prefix + 'ssm_alpha.weight', [Config.LinearHeads, Config.Hidden]);
        Add('linear_attn.in_proj_b.weight', Prefix + 'ssm_beta.weight', [Config.LinearHeads, Config.Hidden]);
        Add('linear_attn.A_log', Prefix + 'ssm_a', [Config.LinearHeads]);
        Add('linear_attn.dt_bias', Prefix + 'ssm_dt.bias', [Config.LinearHeads]);
        Add('linear_attn.conv1d.weight', Prefix + 'ssm_conv1d.weight',
          [ConvWidth, 1, Config.ConvKernel], True);
        Add('linear_attn.norm.weight', Prefix + 'ssm_norm.weight', [Config.LinearValueSize]);
        Add('linear_attn.out_proj.weight', Prefix + 'ssm_out.weight', [Config.Hidden, LinearV]);
      end
      else begin
        Add('self_attn.q_proj.weight', Prefix + 'attn_q.weight', [2 * QWidth, Config.Hidden]);
        Add('self_attn.k_proj.weight', Prefix + 'attn_k.weight', [KVWidth, Config.Hidden]);
        Add('self_attn.v_proj.weight', Prefix + 'attn_v.weight', [KVWidth, Config.Hidden]);
        Add('self_attn.o_proj.weight', Prefix + 'attn_output.weight', [Config.Hidden, QWidth]);
        Add('self_attn.q_norm.weight', Prefix + 'attn_q_norm.weight', [Config.HeadSize]);
        Add('self_attn.k_norm.weight', Prefix + 'attn_k_norm.weight', [Config.HeadSize]);
      end;
    end;
    for I := 0 to Source.Count - 1 do
    begin
      Info := Source.Tensor(I);
      if Copy(Info.Name, 1, 4) = 'blk.' then
      begin
        LayerText := Copy(Info.Name, 5, MaxInt);
        LayerText := Copy(LayerText, 1, Pos('.', LayerText) - 1);
        RequireCpu(TryStrToInt(LayerText, Layer) and (Layer >= 0) and
          (Layer < Config.Layers + Config.MtpLayers), 'Invalid Qwen GGUF tensor layer');
        if Layer < Config.Layers then
          RequireCpu(Used.ContainsKey(Info.Name), 'Unbound Qwen decoder tensor: ' + Info.Name);
      end;
    end;
    SetLength(Bindings, Count);
    Result := Bindings;
  finally
    Used.Free;
  end;
end;

end.
