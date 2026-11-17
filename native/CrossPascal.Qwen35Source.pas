{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35Source                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35Source;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.SafeTensors;

type
  EQwen35Source = class(ESafeTensors);
  TQwen35Attention = (qaGatedDeltaNet, qaFullAttention);
  TQwen35Config = record
    Layers: Integer;
    Hidden: Integer;
    Intermediate: Integer;
    Vocab: Integer;
    Context: Integer;
    Heads: Integer;
    KVHeads: Integer;
    HeadSize: Integer;
    RotaryDimensions: Integer;
    LinearHeads: Integer;
    LinearKVHeads: Integer;
    LinearKeySize: Integer;
    LinearValueSize: Integer;
    ConvKernel: Integer;
    MtpLayers: Integer;
    RMSEpsilon: Double;
    RopeFrequency: Double;
    { Interleaved MRoPE sections (temporal, height, width) in rotary frequencies. }
    MRopeSections: array[0..2] of Integer;
    OutputGateType: string;
    Attention: TArray<TQwen35Attention>;
  end;
  TQwen35Binding = record
    SourceName: string;
    Shape: TArray<Int64>;
  end;

function ReadQwen35Config(const Root: TJSONObject): TQwen35Config;
function ReadQwen35BF16TextBindings(const Source: TSafeTensorCheckpoint;
  const Config: TQwen35Config; out TensorBytes: Int64): TArray<TQwen35Binding>;

implementation

uses
  {$IFDEF FPC}Math, SysUtils, Generics.Collections
  {$ELSE}System.Math, System.SysUtils, System.Generics.Collections{$ENDIF};

procedure RequireQwen(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise EQwen35Source.Create(MessageText);
end;

function ObjectValue(const Root: TJSONObject; const Name: string): TJSONObject;
var
  Value: TJSONValue;
begin
  Value := Root.GetValue(Name);
  RequireQwen(Value is TJSONObject, 'Missing Qwen object: ' + Name);
  Result := TJSONObject(Value);
end;

function TextValue(const Root: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := Root.GetValue(Name);
  RequireQwen(Value is TJSONString, 'Missing Qwen string: ' + Name);
  Result := JSONValueText(Value);
end;

function Dimension(const Root: TJSONObject; const Name: string;
  const Minimum, Maximum: Integer): Integer;
var
  Value: TJSONValue;
  Number: Int64;
begin
  Value := Root.GetValue(Name);
  RequireQwen((Value is TJSONNumber) and TryStrToInt64(JSONValueText(Value), Number),
    'Invalid Qwen integer: ' + Name);
  RequireQwen((Number >= Minimum) and (Number <= Maximum), 'Qwen dimension is out of range: ' + Name);
  Result := Integer(Number);
end;

function PositiveNumber(const Root: TJSONObject; const Name: string): Double;
var
  Value: TJSONValue;
begin
  Value := Root.GetValue(Name);
  RequireQwen((Value is TJSONNumber) and TryStrToFloat(JSONValueText(Value), Result, JSONInvariantFormat),
    'Invalid Qwen number: ' + Name);
  RequireQwen(not IsNan(Result) and not IsInfinite(Result) and (Result > 0),
    'Qwen number must be finite and positive: ' + Name);
end;

procedure BooleanValue(const Root: TJSONObject; const Name: string; const Expected: Boolean);
var
  Value: TJSONValue;
begin
  Value := Root.GetValue(Name);
  RequireQwen(JSONIsBoolean(Value) and (JSONIsTrue(Value) = Expected),
    'Unsupported Qwen boolean: ' + Name);
end;

function ReadQwen35Config(const Root: TJSONObject): TQwen35Config;
var
  Text, Rope: TJSONObject;
  Types, Sections: TJSONArray;
  Value: TJSONValue;
  Partial, Rotary: Double;
  I, Interval, SectionSum: Integer;
begin
  Result := Default(TQwen35Config);
  RequireQwen(Root <> nil, 'A Qwen configuration is required');
  RequireQwen(TextValue(Root, 'model_type') = 'qwen3_5', 'Expected the Qwen 3.5/3.8 architecture');
  Text := ObjectValue(Root, 'text_config');
  RequireQwen(TextValue(Text, 'model_type') = 'qwen3_5_text', 'Expected a dense Qwen text configuration');
  RequireQwen(TextValue(Text, 'dtype') = 'bfloat16', 'This source contract requires original BF16 weights');
  RequireQwen(TextValue(Text, 'hidden_act') = 'silu', 'Unsupported Qwen MLP activation');
  BooleanValue(Text, 'attention_bias', False);
  BooleanValue(Text, 'attn_output_gate', True);
  BooleanValue(Text, 'tie_word_embeddings', False);
  BooleanValue(Text, 'use_cache', True);
  Result.OutputGateType := TextValue(Text, 'output_gate_type');
  RequireQwen(Result.OutputGateType = 'swish', 'Unsupported Qwen output gate type; expected swish');
  Result.Layers := Dimension(Text, 'num_hidden_layers', 1, 1024);
  Result.Hidden := Dimension(Text, 'hidden_size', 1, 65536);
  Result.Intermediate := Dimension(Text, 'intermediate_size', 1, 1048576);
  Result.Vocab := Dimension(Text, 'vocab_size', 1, 16777216);
  Result.Context := Dimension(Text, 'max_position_embeddings', 1, High(Integer));
  Result.Heads := Dimension(Text, 'num_attention_heads', 1, 1024);
  Result.KVHeads := Dimension(Text, 'num_key_value_heads', 1, Result.Heads);
  Result.HeadSize := Dimension(Text, 'head_dim', 1, 65536);
  Result.LinearHeads := Dimension(Text, 'linear_num_value_heads', 1, 1024);
  Result.LinearKVHeads := Dimension(Text, 'linear_num_key_heads', 1, Result.LinearHeads);
  Result.LinearKeySize := Dimension(Text, 'linear_key_head_dim', 1, 65536);
  Result.LinearValueSize := Dimension(Text, 'linear_value_head_dim', 1, 65536);
  Result.ConvKernel := Dimension(Text, 'linear_conv_kernel_dim', 1, 1024);
  Result.MtpLayers := Dimension(Text, 'mtp_num_hidden_layers', 0, 1024);
  Result.RMSEpsilon := PositiveNumber(Text, 'rms_norm_eps');
  RequireQwen((Result.Heads mod Result.KVHeads = 0) and (Result.LinearHeads mod Result.LinearKVHeads = 0),
    'Qwen attention heads do not form equal groups');
  Rope := ObjectValue(Text, 'rope_parameters');
  RequireQwen(TextValue(Rope, 'rope_type') = 'default', 'Unsupported Qwen RoPE scaling');
  BooleanValue(Rope, 'mrope_interleaved', True);
  Result.RopeFrequency := PositiveNumber(Rope, 'rope_theta');
  Partial := PositiveNumber(Rope, 'partial_rotary_factor');
  RequireQwen((Partial <= 1) and (Partial = PositiveNumber(Text, 'partial_rotary_factor')),
    'Inconsistent Qwen partial rotary settings');
  Rotary := Partial * Result.HeadSize;
  RequireQwen((Rotary = Trunc(Rotary)) and (Rotary >= 2), 'Qwen rotary dimension is not a positive integer');
  Result.RotaryDimensions := Trunc(Rotary);
  RequireQwen(Result.RotaryDimensions mod 2 = 0, 'Qwen rotary dimension must be even');
  Value := Rope.GetValue('mrope_section');
  RequireQwen(Value is TJSONArray, 'Missing Qwen MRoPE sections');
  Sections := TJSONArray(Value);
  RequireQwen(Sections.Count = 3, 'Expected three Qwen MRoPE sections');
  SectionSum := 0;
  for I := 0 to Sections.Count - 1 do
  begin
    RequireQwen((Sections.Items[I] is TJSONNumber) and
      TryStrToInt(JSONValueText(Sections.Items[I]), Interval) and
      (Interval >= 0) and (Interval <= Result.HeadSize), 'Invalid Qwen MRoPE section');
    Result.MRopeSections[I] := Interval;
    Inc(SectionSum, Interval);
  end;
  RequireQwen(2 * SectionSum = Result.RotaryDimensions,
    'Qwen MRoPE sections do not cover the rotary dimension');
  Interval := Dimension(Text, 'full_attention_interval', 1, Result.Layers);
  Value := Text.GetValue('layer_types');
  RequireQwen(Value is TJSONArray, 'Missing Qwen layer types');
  Types := TJSONArray(Value);
  RequireQwen(Types.Count = Result.Layers, 'Qwen layer type count differs from its architecture');
  SetLength(Result.Attention, Result.Layers);
  for I := 0 to Result.Layers - 1 do
  begin
    RequireQwen(Types.Items[I] is TJSONString, 'Invalid Qwen layer type');
    if (I + 1) mod Interval = 0 then
    begin
      RequireQwen(JSONValueText(Types.Items[I]) = 'full_attention', 'Qwen full attention pattern differs');
      Result.Attention[I] := qaFullAttention;
    end
    else begin
      RequireQwen(JSONValueText(Types.Items[I]) = 'linear_attention',
        'Qwen linear attention pattern differs');
      Result.Attention[I] := qaGatedDeltaNet;
    end;
  end;
end;

function ReadQwen35BF16TextBindings(const Source: TSafeTensorCheckpoint;
  const Config: TQwen35Config; out TensorBytes: Int64): TArray<TQwen35Binding>;
const
  Prefix = 'model.language_model.';
var
  Bindings: TArray<TQwen35Binding>;
  Names: TDictionary<string, Boolean>;
  Name, LayerPrefix: string;
  Count, Layer: Integer;
  Hidden, Intermediate, QWidth, KVWidth, LinearQK, LinearV: Int64;

  procedure Add(const Name: string; const Shape: array of Int64);
  var
    Info: TSafeTensorInfo;
    I: Integer;
  begin
    Info := Source.TensorInfo(Name);
    RequireQwen((Info.DType = 'BF16') and (Length(Info.Shape) = Length(Shape)),
      'Expected an original BF16 Qwen tensor with the declared rank: ' + Name);
    for I := 0 to High(Shape) do
      RequireQwen(Info.Shape[I] = Shape[I], 'Wrong Qwen tensor shape: ' + Name);
    Bindings[Count].SourceName := Name;
    SetLength(Bindings[Count].Shape, Length(Shape));
    for I := 0 to High(Shape) do Bindings[Count].Shape[I] := Shape[I];
    Names.Add(Name, True);
    RequireQwen(Info.ElementCount <= (High(Int64) - TensorBytes) div 2, 'Qwen tensor byte total overflow');
    Inc(TensorBytes, 2 * Info.ElementCount);
    Inc(Count);
  end;

begin
  Result := nil;
  TensorBytes := 0;
  RequireQwen((Source <> nil) and (Config.Layers > 0) and (Length(Config.Attention) = Config.Layers),
    'A validated Qwen source and configuration are required');
  SetLength(Bindings, Config.Layers * 14 + 3);
  Names := TDictionary<string, Boolean>.Create;
  try
    Count := 0;
    Hidden := Config.Hidden;
    Intermediate := Config.Intermediate;
    QWidth := Int64(Config.Heads) * Config.HeadSize;
    KVWidth := Int64(Config.KVHeads) * Config.HeadSize;
    LinearQK := Int64(Config.LinearKVHeads) * Config.LinearKeySize;
    LinearV := Int64(Config.LinearHeads) * Config.LinearValueSize;
    Add(Prefix + 'embed_tokens.weight', [Config.Vocab, Hidden]);
    for Layer := 0 to Config.Layers - 1 do
    begin
      LayerPrefix := Prefix + 'layers.' + IntToStr(Layer) + '.';
      Add(LayerPrefix + 'input_layernorm.weight', [Hidden]);
      Add(LayerPrefix + 'post_attention_layernorm.weight', [Hidden]);
      Add(LayerPrefix + 'mlp.gate_proj.weight', [Intermediate, Hidden]);
      Add(LayerPrefix + 'mlp.up_proj.weight', [Intermediate, Hidden]);
      Add(LayerPrefix + 'mlp.down_proj.weight', [Hidden, Intermediate]);
      if Config.Attention[Layer] = qaGatedDeltaNet then
      begin
        Add(LayerPrefix + 'linear_attn.A_log', [Config.LinearHeads]);
        Add(LayerPrefix + 'linear_attn.dt_bias', [Config.LinearHeads]);
        Add(LayerPrefix + 'linear_attn.conv1d.weight', [2 * LinearQK + LinearV, 1, Config.ConvKernel]);
        Add(LayerPrefix + 'linear_attn.in_proj_a.weight', [Config.LinearHeads, Hidden]);
        Add(LayerPrefix + 'linear_attn.in_proj_b.weight', [Config.LinearHeads, Hidden]);
        Add(LayerPrefix + 'linear_attn.in_proj_qkv.weight', [2 * LinearQK + LinearV, Hidden]);
        Add(LayerPrefix + 'linear_attn.in_proj_z.weight', [LinearV, Hidden]);
        Add(LayerPrefix + 'linear_attn.norm.weight', [Config.LinearValueSize]);
        Add(LayerPrefix + 'linear_attn.out_proj.weight', [Hidden, LinearV]);
      end
      else begin
        Add(LayerPrefix + 'self_attn.q_proj.weight', [2 * QWidth, Hidden]);
        Add(LayerPrefix + 'self_attn.k_proj.weight', [KVWidth, Hidden]);
        Add(LayerPrefix + 'self_attn.v_proj.weight', [KVWidth, Hidden]);
        Add(LayerPrefix + 'self_attn.o_proj.weight', [Hidden, QWidth]);
        Add(LayerPrefix + 'self_attn.q_norm.weight', [Config.HeadSize]);
        Add(LayerPrefix + 'self_attn.k_norm.weight', [Config.HeadSize]);
      end;
    end;
    Add(Prefix + 'norm.weight', [Hidden]);
    Add('lm_head.weight', [Config.Vocab, Hidden]);
    for Name in Source.TensorNames do
      if Copy(Name, 1, Length(Prefix)) = Prefix then
        RequireQwen(Names.ContainsKey(Name), 'Unbound Qwen text tensor: ' + Name);
    SetLength(Bindings, Count);
    Result := Bindings;
  finally
    Names.Free;
  end;
end;

end.
