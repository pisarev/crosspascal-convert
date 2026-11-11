{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen2Graph                                                     }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen2Graph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Onnx, CrossPascal.SafeTensors;

type
  TQwen2Config = record
    Hidden: Integer;
    Intermediate: Integer;
    Heads: Integer;
    KVHeads: Integer;
    HeadSize: Integer;
    Layers: Integer;
    Vocab: Integer;
    MaxContext: Integer;
    Epsilon: Single;
    RotaryBase: Single;
    TieEmbeddings: Boolean;
  end;

function ReadQwen2Config(const Config: TJSONObject): TQwen2Config;
function BuildQwen2Graph(const Config: TQwen2Config; const Checkpoint: TSafeTensorCheckpoint;
  const Context, Chunk: Integer): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Generics.Collections{$ELSE}System.SysUtils, System.Math, System.Generics.Collections{$ENDIF},
  CrossPascal.ModelGraph, TVM.CpuGraph;

type
  TQwen2GraphBuilder = class(TModelGraphBuilder)
  private
    FConfig: TQwen2Config;
    FCheckpoint: TSafeTensorCheckpoint;
    FWeights: TDictionary<string, Byte>;
    FContext: Integer;
    FChunk: Integer;
    FCacheShape: TArray<Int64>;
    FRotary: TRotaryEmbedding;
    FAttentionMask: string;
    FCacheMask: string;
    FCacheIndex: string;
    function Weight(const Name: string; const Shape: array of Int64): string;
    function Linear(const Input, Name: string; const Columns, Rows: Integer;
      const Bias: Boolean = False): string;
    function Normalize(const Input, Name: string): string;
    function Attention(const Input, Prefix: string; const Layer: Integer): string;
    function DecoderLayer(const Input: string; const Index: Integer): string;
    procedure AddMasks;
  public
    constructor Create(const Config: TQwen2Config; const Checkpoint: TSafeTensorCheckpoint;
      const Context, Chunk: Integer);
    destructor Destroy; override;
    function Build: TOnnxModel;
  end;

function ReadQwen2Config(const Config: TJSONObject): TQwen2Config;

  function Dimension(const Name: string): Integer;
  var
    Value: Int64;
  begin
    Value := ModelInt(Config.GetValue(Name), Name);
    RequireCpu((Value > 0) and (Value <= 1048576), 'Invalid Qwen2 dimension: ' + Name);
    Result := Value;
  end;

var
  Scaling: TJSONValue;
begin
  Scaling := Config.GetValue('rope_scaling');
  RequireCpu(
    (ModelText(Config.GetValue('model_type')) = 'qwen2') and (ModelText(Config.GetValue('hidden_act')) = 'silu') and not JSONIsTrue(Config.GetValue('use_sliding_window')) and ((Scaling = nil) or (Scaling is TJSONNull)),
    'Expected dense Qwen2 with standard rotary positions'
  );
  Result.Hidden := Dimension('hidden_size');
  Result.Intermediate := Dimension('intermediate_size');
  Result.Heads := Dimension('num_attention_heads');
  Result.KVHeads := Dimension('num_key_value_heads');
  Result.Layers := Dimension('num_hidden_layers');
  Result.Vocab := Dimension('vocab_size');
  Result.MaxContext := Dimension('max_position_embeddings');
  RequireCpu((Result.Hidden mod Result.Heads = 0) and (Result.Heads mod Result.KVHeads = 0) and
    (Result.Layers <= 128) and (Result.MaxContext <= 131072), 'Invalid Qwen2 attention dimensions');
  Result.HeadSize := Result.Hidden div Result.Heads;
  RequireCpu(Result.HeadSize mod 2 = 0, 'Qwen2 head size must be even');
  Result.Epsilon := ModelFloat(Config.GetValue('rms_norm_eps'), 'rms_norm_eps');
  Result.RotaryBase := ModelFloat(Config.GetValue('rope_theta'), 'rope_theta');
  RequireCpu((Result.Epsilon > 0) and (Result.Epsilon < 1) and (Result.RotaryBase >= 1) and
    not IsInfinite(Result.RotaryBase), 'Invalid Qwen2 normalization or rotary base');
  Result.TieEmbeddings := JSONIsTrue(Config.GetValue('tie_word_embeddings'));
end;

constructor TQwen2GraphBuilder.Create(const Config: TQwen2Config;
  const Checkpoint: TSafeTensorCheckpoint; const Context, Chunk: Integer);
begin
  inherited Create;
  RequireCpu((Context >= 2) and (Context <= Config.MaxContext) and (Chunk >= 1) and
    (Chunk <= Context), 'Invalid Qwen2 graph context');
  FConfig := Config;
  FCheckpoint := Checkpoint;
  FContext := Context;
  FChunk := Chunk;
  FCacheShape := [1, Config.KVHeads, Context, Config.HeadSize];
  FWeights := TDictionary<string, Byte>.Create;
end;

destructor TQwen2GraphBuilder.Destroy;
begin
  FWeights.Free;
  inherited;
end;

function TQwen2GraphBuilder.Weight(const Name: string; const Shape: array of Int64): string;
var
  Info: TSafeTensorInfo;
  I: Integer;
begin
  Result := Name;
  if FWeights.ContainsKey(Name) then Exit;
  Info := FCheckpoint.TensorInfo(Name);
  RequireCpu((Length(Info.Shape) = Length(Shape)) and
    ((Info.DType = 'BF16') or (Info.DType = 'F16') or (Info.DType = 'F32')),
    'Invalid Qwen2 checkpoint tensor: ' + Name);
  for I := 0 to High(Shape) do
    RequireCpu(Info.Shape[I] = Shape[I], 'Invalid Qwen2 tensor shape: ' + Name);
  FInputs.Add(ValueInfo(Name, 1, Shape));
  FWeights.Add(Name, 0);
end;

function TQwen2GraphBuilder.Linear(const Input, Name: string; const Columns, Rows: Integer;
  const Bias: Boolean): string;
begin
  Result := Op('MatMul', [Input, Transpose(Weight(Name + '.weight', [Rows, Columns]), [1, 0])]);
  if Bias then Result := Op('Add', [Result, Weight(Name + '.bias', [Rows])]);
end;

function TQwen2GraphBuilder.Normalize(const Input, Name: string): string;
begin
  Result := Op('Mul', [RMSNormalize(Input, FConfig.Epsilon), Weight(Name, [FConfig.Hidden])]);
end;

procedure TQwen2GraphBuilder.AddMasks;
var
  Positions: TArray<Int64>;
  I: Integer;
  Past, Distance, Valid: string;
begin
  SetLength(Positions, FContext);
  for I := 0 to High(Positions) do Positions[I] := I;
  Past := Integers(Positions);
  Distance := Op('Sub', [Reshape('position', [FChunk, 1]), Reshape(Past, [1, FContext])]);
  Valid := Op('GreaterOrEqual', [Distance, IntegerValue(0)]);
  FAttentionMask := Op('Where', [Valid, Float(0), Float(-3.402823466e38)]);
  if FChunk = 1 then
    Valid := Op('Equal', [Past, 'position'])
  else begin
    FCacheIndex := Op('Sub', [Past, Gather('position', Integers([0]), 0)]);
    Valid := Op('And', [Op('GreaterOrEqual', [FCacheIndex, IntegerValue(0)]),
      Op('Less', [FCacheIndex, IntegerValue(FChunk)])]);
    FCacheIndex := Op('Where', [Valid, FCacheIndex, IntegerValue(0)]);
  end;
  FCacheMask := Reshape(Valid, [1, 1, FContext, 1]);
end;

function TQwen2GraphBuilder.Attention(const Input, Prefix: string; const Layer: Integer): string;
var
  Queries, Key, Value, Scores, Hidden, KeyID, ValueID: string;
  KVRows: Integer;
begin
  KVRows := FConfig.KVHeads * FConfig.HeadSize;
  Queries := Reshape(Linear(Input, Prefix + 'q_proj', FConfig.Hidden, FConfig.Hidden, True),
    [1, FChunk, FConfig.Heads, FConfig.HeadSize]);
  Key := Reshape(Linear(Input, Prefix + 'k_proj', FConfig.Hidden, KVRows, True),
    [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]);
  Value := Reshape(Linear(Input, Prefix + 'v_proj', FConfig.Hidden, KVRows, True),
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
  AddOutput('updated_' + KeyID, Key, 1, FCacheShape);
  AddOutput('updated_' + ValueID, Value, 1, FCacheShape);
  Key := Transpose(RepeatHeads(Key, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize),
    [0, 1, 3, 2]);
  Value := RepeatHeads(Value, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize);
  Scores := Op('Mul', [Op('MatMul', [Queries, Key]), Float(1 / Sqrt(FConfig.HeadSize))]);
  Scores := AxisOp('Softmax', [Op('Add', [Scores, FAttentionMask])], -1);
  Hidden := Transpose(Op('MatMul', [Scores, Value]), [0, 2, 1, 3]);
  Result := Linear(Reshape(Hidden, [1, FChunk, FConfig.Hidden]), Prefix + 'o_proj',
    FConfig.Hidden, FConfig.Hidden);
end;

function TQwen2GraphBuilder.DecoderLayer(const Input: string; const Index: Integer): string;
var
  Prefix, Hidden, Residual, Gate, Up: string;
begin
  Prefix := 'model.layers.' + IntToStr(Index) + '.';
  Hidden := Attention(Normalize(Input, Prefix + 'input_layernorm.weight'), Prefix + 'self_attn.', Index);
  Residual := Op('Add', [Input, Hidden]);
  Hidden := Normalize(Residual, Prefix + 'post_attention_layernorm.weight');
  Gate := Linear(Hidden, Prefix + 'mlp.gate_proj', FConfig.Hidden, FConfig.Intermediate);
  Gate := Op('Mul', [Gate, Op('Sigmoid', [Gate])]);
  Up := Linear(Hidden, Prefix + 'mlp.up_proj', FConfig.Hidden, FConfig.Intermediate);
  Hidden := Linear(Op('Mul', [Gate, Up]), Prefix + 'mlp.down_proj', FConfig.Intermediate, FConfig.Hidden);
  Result := Op('Add', [Residual, Hidden]);
end;

function TQwen2GraphBuilder.Build: TOnnxModel;
var
  I: Integer;
  Hidden, Head, Logits: string;
  LogitsOutput: TOnnxValue;
begin
  FInputs.Add(ValueInfo('token', 7, [1, FChunk]));
  FInputs.Add(ValueInfo('position', 7, [FChunk]));
  for I := 0 to FConfig.Layers * 2 - 1 do
    FInputs.Add(ValueInfo('cache_' + IntToStr(I), 1, FCacheShape));
  AddMasks;
  FRotary := Rotary('position', FConfig.RotaryBase, FChunk, FConfig.HeadSize);
  Hidden := Gather(Weight('model.embed_tokens.weight', [FConfig.Vocab, FConfig.Hidden]), 'token', 0);
  for I := 0 to FConfig.Layers - 1 do Hidden := DecoderLayer(Hidden, I);
  Hidden := Normalize(Gather(Hidden, IntegerValue(FChunk - 1), 1), 'model.norm.weight');
  if FConfig.TieEmbeddings then
    Head := 'model.embed_tokens'
  else
    Head := 'lm_head';
  Logits := Linear(Hidden, Head, FConfig.Hidden, FConfig.Vocab);
  AddOutput('logits', Logits, 1, [1, FConfig.Vocab]);
  LogitsOutput := FOutputs.Last;
  FOutputs.Delete(FOutputs.Count - 1);
  FOutputs.Insert(0, LogitsOutput);
  Result := CreateModel;
end;

function BuildQwen2Graph(const Config: TQwen2Config; const Checkpoint: TSafeTensorCheckpoint;
  const Context, Chunk: Integer): TOnnxModel;
var
  Builder: TQwen2GraphBuilder;
begin
  Builder := TQwen2GraphBuilder.Create(Config, Checkpoint, Context, Chunk);
  try
    Result := Builder.Build;
  finally
    Builder.Free;
  end;
end;

end.
