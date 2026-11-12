{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma2Graph                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma2Graph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Onnx, CrossPascal.SafeTensors;

type
  TGemma2Config = record
    Hidden: Integer;
    Intermediate: Integer;
    Heads: Integer;
    KVHeads: Integer;
    HeadSize: Integer;
    Layers: Integer;
    Vocab: Integer;
    MaxContext: Integer;
    Window: Integer;
    RMS: Single;
    RotaryBase: Single;
    QueryScale: Single;
    AttentionCap: Single;
    OutputCap: Single;
    TieEmbeddings: Boolean;
  end;

function ReadGemma2Config(const Config: TJSONObject): TGemma2Config;
function BuildGemma2Graph(const Config: TGemma2Config; const Weights: TSafeTensorCheckpoint;
  const Context, Chunk: Integer; const Quantize: Boolean = False): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Generics.Collections{$ELSE}System.SysUtils, System.Math, System.Generics.Collections{$ENDIF},
  CrossPascal.ModelGraph, TVM.CpuGraph;

type
  TGemma2GraphBuilder = class(TModelGraphBuilder)
  private
    FConfig: TGemma2Config;
    FCheckpoint: TSafeTensorCheckpoint;
    FKnownWeights: TDictionary<string, Byte>;
    FContext: Integer;
    FChunk: Integer;
    FQuantize: Boolean;
    FRotary: TRotaryEmbedding;
    FCacheMask: string;
    FCacheIndex: string;
    FFullMask: string;
    FLocalMask: string;
    function Weight(const Name: string; const Shape: array of Int64): string;
    function Linear(const Input, Name: string; const Features, Rows: Integer): string;
    function Normalize(const Input, Name: string): string;
    function SoftCap(const Input: string; const Cap: Single): string;
    procedure AddMasks;
    function Attention(const Input, Prefix: string; const Layer: Integer): string;
    function DecoderLayer(const Input: string; const Index: Integer): string;
  public
    constructor Create(const Config: TGemma2Config; const Checkpoint: TSafeTensorCheckpoint;
      const Context, Chunk: Integer; const Quantize: Boolean);
    destructor Destroy; override;
    function Build: TOnnxModel;
  end;

function ReadGemma2Config(const Config: TJSONObject): TGemma2Config;

  function Dimension(const Name: string): Integer;
  var
    Value: Int64;
  begin
    Value := ModelInt(Config.GetValue(Name), Name);
    RequireCpu((Value > 0) and (Value <= 1048576), 'Invalid Gemma 2 dimension: ' + Name);
    Result := Value;
  end;

  function Positive(const Name: string): Single;
  var
    Value: Double;
  begin
    Value := ModelFloat(Config.GetValue(Name), Name);
    RequireCpu((Value > 0) and (Value < 1e30), 'Invalid Gemma 2 setting: ' + Name);
    Result := Value;
  end;

var
  Tied: TJSONValue;
begin
  RequireCpu((ModelText(Config.GetValue('model_type')) = 'gemma2') and
    (ModelText(Config.GetValue('hidden_activation')) = 'gelu_pytorch_tanh') and
    not JSONIsTrue(Config.GetValue('attention_bias')), 'Unsupported Gemma 2 architecture');
  Result.Hidden := Dimension('hidden_size');
  Result.Intermediate := Dimension('intermediate_size');
  Result.Heads := Dimension('num_attention_heads');
  Result.KVHeads := Dimension('num_key_value_heads');
  Result.HeadSize := Dimension('head_dim');
  Result.Layers := Dimension('num_hidden_layers');
  Result.Vocab := Dimension('vocab_size');
  Result.MaxContext := Dimension('max_position_embeddings');
  Result.Window := Dimension('sliding_window');
  Result.RMS := Positive('rms_norm_eps');
  Result.RotaryBase := Positive('rope_theta');
  Result.QueryScale := 1 / Sqrt(Positive('query_pre_attn_scalar'));
  Result.AttentionCap := Positive('attn_logit_softcapping');
  Result.OutputCap := Positive('final_logit_softcapping');
  Tied := Config.GetValue('tie_word_embeddings');
  RequireCpu((Tied = nil) or JSONIsBoolean(Tied), 'Invalid Gemma 2 embedding setting');
  Result.TieEmbeddings := (Tied = nil) or JSONIsTrue(Tied);
  RequireCpu((Result.Heads mod Result.KVHeads = 0) and (Result.HeadSize mod 2 = 0) and
    (Int64(Result.Heads) * Result.HeadSize <= MaxInt) and
    (Result.Layers <= 128) and (Result.MaxContext <= 131072), 'Invalid Gemma 2 dimensions');
end;

constructor TGemma2GraphBuilder.Create(const Config: TGemma2Config;
  const Checkpoint: TSafeTensorCheckpoint; const Context, Chunk: Integer; const Quantize: Boolean);
begin
  inherited Create;
  RequireCpu((Context >= 2) and (Context <= Config.MaxContext) and (Chunk >= 1) and
    (Chunk <= Context), 'Invalid Gemma 2 graph context');
  FConfig := Config;
  FCheckpoint := Checkpoint;
  FContext := Context;
  FChunk := Chunk;
  FQuantize := Quantize;
  FKnownWeights := TDictionary<string, Byte>.Create;
end;

destructor TGemma2GraphBuilder.Destroy;
begin
  FKnownWeights.Free;
  inherited;
end;

function TGemma2GraphBuilder.Weight(const Name: string; const Shape: array of Int64): string;
var
  Info: TSafeTensorInfo;
  I: Integer;
begin
  Result := Name;
  if FKnownWeights.ContainsKey(Name) then Exit;
  Info := FCheckpoint.TensorInfo(Name);
  RequireCpu((Length(Info.Shape) = Length(Shape)) and
    ((Info.DType = 'BF16') or (Info.DType = 'F16') or (Info.DType = 'F32')),
    'Invalid Gemma 2 checkpoint tensor: ' + Name);
  for I := 0 to High(Shape) do
    RequireCpu(Info.Shape[I] = Shape[I], 'Invalid Gemma 2 tensor shape: ' + Name);
  FInputs.Add(ValueInfo(Name, 1, Shape));
  FKnownWeights.Add(Name, 0);
end;

function TGemma2GraphBuilder.Linear(const Input, Name: string; const Features, Rows: Integer): string;
var
  Node: TOnnxNode;
  Quantized, Scale: string;
  Info: TSafeTensorInfo;
begin
  if not FQuantize then
    Exit(Op('MatMul', [Input, Transpose(Weight(Name, [Rows, Features]), [1, 0])]));
  Quantized := Name + ':int8';
  Scale := Name + ':scale';
  if not FKnownWeights.ContainsKey(Quantized) then
  begin
    Info := FCheckpoint.TensorInfo(Name);
    RequireCpu((Length(Info.Shape) = 2) and (Info.Shape[0] = Rows) and
      (Info.Shape[1] = Features), 'Invalid Gemma 2 matrix shape: ' + Name);
    FInputs.Add(ValueInfo(Quantized, OnnxInt8, [Rows, Features]));
    FInputs.Add(ValueInfo(Scale, OnnxFloat, [Rows]));
    FKnownWeights.Add(Quantized, 0);
  end;
  Result := NextName;
  Node := Default(TOnnxNode);
  Node.Name := Result;
  Node.Domain := 'crosspascal';
  Node.OpType := 'Q8MatMul';
  Node.Inputs := TArray<string>.Create(Input, Quantized, Scale);
  Node.Outputs := TArray<string>.Create(Result);
  FNodes.Add(Node);
end;

function TGemma2GraphBuilder.Normalize(const Input, Name: string): string;
begin
  Result := Op('Mul', [RMSNormalize(Input, FConfig.RMS),
    Op('Add', [Float(1), Weight(Name, [FConfig.Hidden])])]);
end;

function TGemma2GraphBuilder.SoftCap(const Input: string; const Cap: Single): string;
begin
  Result := Op('Mul', [Op('Tanh', [Op('Mul', [Input, Float(1 / Cap)])]), Float(Cap)]);
end;

procedure TGemma2GraphBuilder.AddMasks;
var
  Positions: TArray<Int64>;
  I: Integer;
  Past, Relative, Valid, Mask, Lowest, Zero: string;
begin
  SetLength(Positions, FContext);
  for I := 0 to High(Positions) do Positions[I] := I;
  Past := Integers(Positions);
  Relative := Op('Sub', [Reshape('position', [FChunk, 1]), Reshape(Past, [1, FContext])]);
  Valid := Op('GreaterOrEqual', [Relative, IntegerValue(0)]);
  Lowest := Float(-3.402823466e38);
  Zero := Float(0);
  FFullMask := Op('Where', [Valid, Zero, Lowest]);
  FLocalMask := FFullMask;
  if FConfig.Window < FContext then
  begin
    Mask := Op('And', [Valid, Op('Less', [Relative, IntegerValue(FConfig.Window)])]);
    FLocalMask := Op('Where', [Mask, Zero, Lowest]);
  end;
  if FChunk = 1 then
    Mask := Op('Equal', [Past, 'position'])
  else begin
    FCacheIndex := Op('Sub', [Past, Gather('position', IntegerValue(0), 0)]);
    Mask := Op('And', [Op('GreaterOrEqual', [FCacheIndex, IntegerValue(0)]),
      Op('Less', [FCacheIndex, IntegerValue(FChunk)])]);
    FCacheIndex := Op('Where', [Mask, FCacheIndex, IntegerValue(0)]);
  end;
  FCacheMask := Reshape(Mask, [1, 1, FContext, 1]);
end;

function TGemma2GraphBuilder.Attention(const Input, Prefix: string; const Layer: Integer): string;
var
  Queries, Key, Value, Scores, Hidden, Mask, KeyID, ValueID: string;
  Rows, KVRows: Integer;
begin
  Rows := FConfig.Heads * FConfig.HeadSize;
  KVRows := FConfig.KVHeads * FConfig.HeadSize;
  Queries := Reshape(Linear(Input, Prefix + 'q_proj.weight', FConfig.Hidden, Rows),
    [1, FChunk, FConfig.Heads, FConfig.HeadSize]);
  Key := Reshape(Linear(Input, Prefix + 'k_proj.weight', FConfig.Hidden, KVRows),
    [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]);
  Value := Reshape(Linear(Input, Prefix + 'v_proj.weight', FConfig.Hidden, KVRows),
    [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]);
  Queries := Transpose(ApplyRotary(Queries, FRotary, FConfig.HeadSize), [0, 2, 1, 3]);
  Key := Transpose(ApplyRotary(Key, FRotary, FConfig.HeadSize), [0, 2, 1, 3]);
  Value := Transpose(Value, [0, 2, 1, 3]);
  if FChunk > 1 then
  begin
    Key := Gather(Key, FCacheIndex, 2);
    Value := Gather(Value, FCacheIndex, 2);
  end;
  KeyID := IntToStr(2 * Layer);
  ValueID := IntToStr(2 * Layer + 1);
  Key := Op('Where', [FCacheMask, Key, 'cache_' + KeyID]);
  Value := Op('Where', [FCacheMask, Value, 'cache_' + ValueID]);
  AddOutput('updated_' + KeyID, Key, 1, [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  AddOutput('updated_' + ValueID, Value, 1, [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  Key := Transpose(RepeatHeads(Key, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize),
    [0, 1, 3, 2]);
  Value := RepeatHeads(Value, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize);
  Scores := Op('Mul', [Op('MatMul', [Queries, Key]), Float(FConfig.QueryScale)]);
  Scores := SoftCap(Scores, FConfig.AttentionCap);
  if Odd(Layer) then
    Mask := FFullMask
  else
    Mask := FLocalMask;
  Scores := AxisOp('Softmax', [Op('Add', [Scores, Mask])], -1);
  Hidden := Transpose(Op('MatMul', [Scores, Value]), [0, 2, 1, 3]);
  Result := Linear(Reshape(Hidden, [1, FChunk, Rows]), Prefix + 'o_proj.weight', Rows, FConfig.Hidden);
end;

function TGemma2GraphBuilder.DecoderLayer(const Input: string; const Index: Integer): string;
var
  Prefix, Hidden, Residual, Gate, Up: string;
begin
  Prefix := 'model.layers.' + IntToStr(Index) + '.';
  Hidden := Attention(Normalize(Input, Prefix + 'input_layernorm.weight'), Prefix + 'self_attn.', Index);
  Residual := Op('Add', [Input, Normalize(Hidden, Prefix + 'post_attention_layernorm.weight')]);
  Hidden := Normalize(Residual, Prefix + 'pre_feedforward_layernorm.weight');
  Gate := GELU(Linear(Hidden, Prefix + 'mlp.gate_proj.weight', FConfig.Hidden, FConfig.Intermediate));
  Up := Linear(Hidden, Prefix + 'mlp.up_proj.weight', FConfig.Hidden, FConfig.Intermediate);
  Hidden := Linear(Op('Mul', [Gate, Up]), Prefix + 'mlp.down_proj.weight', FConfig.Intermediate,
    FConfig.Hidden);
  Result := Op('Add', [Residual, Normalize(Hidden, Prefix + 'post_feedforward_layernorm.weight')]);
end;

function TGemma2GraphBuilder.Build: TOnnxModel;
var
  I: Integer;
  Hidden, TableName, Head, Logits: string;
  LogitsOutput: TOnnxValue;
begin
  FInputs.Add(ValueInfo('token', 7, [1, FChunk]));
  FInputs.Add(ValueInfo('position', 7, [FChunk]));
  for I := 0 to FConfig.Layers * 2 - 1 do
    FInputs.Add(ValueInfo('cache_' + IntToStr(I), 1, [1, FConfig.KVHeads, FContext, FConfig.HeadSize]));
  AddMasks;
  FRotary := Rotary('position', FConfig.RotaryBase, FChunk, FConfig.HeadSize);
  TableName := 'model.embed_tokens.weight';
  Hidden := Gather(Weight(TableName, [FConfig.Vocab, FConfig.Hidden]), 'token', 0);
  Hidden := Op('Mul', [Hidden, Float(Sqrt(FConfig.Hidden))]);
  for I := 0 to FConfig.Layers - 1 do Hidden := DecoderLayer(Hidden, I);
  Hidden := Normalize(Gather(Hidden, IntegerValue(FChunk - 1), 1), 'model.norm.weight');
  if FConfig.TieEmbeddings then
    Head := TableName
  else
    Head := 'lm_head.weight';
  Logits := SoftCap(Linear(Hidden, Head, FConfig.Hidden, FConfig.Vocab), FConfig.OutputCap);
  AddOutput('logits', Logits, 1, [1, FConfig.Vocab]);
  LogitsOutput := FOutputs.Last;
  FOutputs.Delete(FOutputs.Count - 1);
  FOutputs.Insert(0, LogitsOutput);
  Result := CreateModel;
end;

function BuildGemma2Graph(const Config: TGemma2Config; const Weights: TSafeTensorCheckpoint;
  const Context, Chunk: Integer; const Quantize: Boolean): TOnnxModel;
var
  Builder: TGemma2GraphBuilder;
begin
  Builder := TGemma2GraphBuilder.Create(Config, Weights, Context, Chunk, Quantize);
  try
    Result := Builder.Build;
  finally
    Builder.Free;
  end;
end;

end.
