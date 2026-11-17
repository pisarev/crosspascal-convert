{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35Graph                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35Graph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Onnx, CrossPascal.Qwen35Source;

type
  TQwen35WeightConvention = (qwOriginal, qwGGUF);
  TQwen35WeightStorage = (qwsFloat32, qwsBF16, qwsFloat32Row);

{ One decoder layer with explicit recurrent/cache state, FP32 activations and
  FP32 or BF16 parameter storage. Layer-local names let a streamed runtime reuse
  the same compiled graph for every layer of the same type. }
function BuildQwen35LayerGraph(const Config: TQwen35Config; const Kind: TQwen35Attention;
  const Context: Integer; const Convention: TQwen35WeightConvention;
  const Storage: TQwen35WeightStorage = qwsFloat32): TOnnxModel;
function BuildQwen35HeadGraph(const Config: TQwen35Config; const Rows: Integer;
  const Convention: TQwen35WeightConvention;
  const Storage: TQwen35WeightStorage = qwsFloat32): TOnnxModel;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, CrossPascal.ModelGraph,
  CrossPascal.GatedDeltaNetGraph, TVM.CpuGraph;

type
  TQwen35LayerBuilder = class(TGatedDeltaNetGraphBuilder)
  private
    FConfig: TQwen35Config;
    FContext: Integer;
    FConvention: TQwen35WeightConvention;
    FStorage: TQwen35WeightStorage;
    function Input(const Name: string; const Shape: array of Int64;
      const DataType: Integer = OnnxFloat): string;
    function Parameter(const Name: string; const Shape: array of Int64;
      const KeepBF16: Boolean = False): string;
    function Linear(const X, Name: string; const Columns, Rows: Integer): string;
    function Normalize(const X, Name: string; const Width: Integer;
      const ZeroCentered: Boolean = True): string;
    function Silu(const X: string): string;
    function Softplus(const X: string): string;
    function LinearAttention(const X: string): string;
    function FullAttention(const X: string): string;
    function PartialRotary(const X: string; const Factors: TRotaryEmbedding): string;
  public
    function Build(const Config: TQwen35Config; const Kind: TQwen35Attention;
      const Context: Integer; const Convention: TQwen35WeightConvention;
      const Storage: TQwen35WeightStorage): TOnnxModel;
    function BuildHead(const Config: TQwen35Config; const Rows: Integer;
      const Convention: TQwen35WeightConvention; const Storage: TQwen35WeightStorage): TOnnxModel;
  end;

{ Transformers Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope: every
  frequency starts on the temporal axis; for axis 1 (height) and 2 (width),
  frequencies Axis, Axis + 3, ... below 3 * section[Axis] take that axis. }
function Qwen35MRopeSelector(const Config: TQwen35Config): TArray<Int64>;
var
  Half, J, Axis: Integer;
begin
  Half := Config.RotaryDimensions div 2;
  RequireCpu((Half > 0) and (Config.MRopeSections[0] >= 0) and (Config.MRopeSections[1] >= 0) and
    (Config.MRopeSections[2] >= 0) and
    (Config.MRopeSections[0] + Config.MRopeSections[1] + Config.MRopeSections[2] = Half),
    'Qwen MRoPE sections do not cover the rotary frequencies');
  SetLength(Result, Half);
  for J := 0 to Half - 1 do
  begin
    Result[J] := 0;
    for Axis := 1 to 2 do
      if (J mod 3 = Axis) and (J < 3 * Config.MRopeSections[Axis]) then Result[J] := Axis;
  end;
end;

function TQwen35LayerBuilder.Input(const Name: string; const Shape: array of Int64;
  const DataType: Integer): string;
begin
  FInputs.Add(ValueInfo(Name, DataType, Shape));
  Result := Name;
end;

function TQwen35LayerBuilder.Parameter(const Name: string; const Shape: array of Int64;
  const KeepBF16: Boolean): string;
begin
  if FStorage = qwsBF16 then
  begin
    Result := Input(Name, Shape, OnnxBFloat16);
    if not KeepBF16 then Result := Cast(Result, OnnxFloat);
  end
  else
    Result := Input(Name, Shape);
end;

function TQwen35LayerBuilder.Linear(const X, Name: string; const Columns, Rows: Integer): string;
var
  W: string;
  Node: TOnnxNode;
begin
  W := Parameter(Name + '.weight', [Rows, Columns], True);
  if FStorage in [qwsBF16, qwsFloat32Row] then
  begin
    if FStorage = qwsBF16 then
      Result := Op('BF16MatMul', [X, W])
    else
      Result := Op('Float32RowMatMul', [X, W]);
    Node := FNodes.Last;
    Node.Domain := 'crosspascal';
    FNodes[FNodes.Count - 1] := Node;
  end
  else
    Result := Op('MatMul', [X, Transpose(W, [1, 0])]);
end;

function TQwen35LayerBuilder.Normalize(const X, Name: string; const Width: Integer;
  const ZeroCentered: Boolean): string;
var
  W: string;
begin
  W := Parameter(Name + '.weight', [Width]);
  if ZeroCentered and (FConvention = qwOriginal) then W := Op('Add', [W, Float(1)]);
  Result := Op('Mul', [RMSNormalize(X, FConfig.RMSEpsilon), W]);
end;

function TQwen35LayerBuilder.Silu(const X: string): string;
begin
  Result := Op('Mul', [X, Op('Sigmoid', [X])]);
end;

function TQwen35LayerBuilder.Softplus(const X: string): string;
var
  Absolute: string;
begin
  Absolute := Op('Max', [X, Op('Neg', [X])]);
  Result := Op('Add', [Op('Max', [X, Float(0)]),
    Op('Log', [Op('Add', [Float(1), Op('Exp', [Op('Neg', [Absolute])])])])]);
end;

function TQwen35LayerBuilder.LinearAttention(const X: string): string;
const
  Prefix = 'linear_attn.';
var
  QKWidth, VWidth, ConvWidth: Integer;
  Mixed, Conv, NewConv, ConvWeight, Query, Key, Value, Z, A, B, Decay: string;
  State, Updated, Output, Projected, NegativeA: string;
begin
  QKWidth := FConfig.LinearKVHeads * FConfig.LinearKeySize;
  VWidth := FConfig.LinearHeads * FConfig.LinearValueSize;
  ConvWidth := QKWidth * 2 + VWidth;
  Mixed := Linear(X, Prefix + 'in_proj_qkv', FConfig.Hidden, ConvWidth);
  Z := Linear(X, Prefix + 'in_proj_z', FConfig.Hidden, VWidth);
  A := Linear(X, Prefix + 'in_proj_a', FConfig.Hidden, FConfig.LinearHeads);
  B := Linear(X, Prefix + 'in_proj_b', FConfig.Hidden, FConfig.LinearHeads);
  Conv := Input('conv_state', [1, ConvWidth, FConfig.ConvKernel]);
  Projected := Transpose(Mixed, [0, 2, 1]);
  if FConfig.ConvKernel = 1 then
    NewConv := Projected
  else
    NewConv := AxisOp('Concat', [Part(Conv, 2, 1, FConfig.ConvKernel), Projected], 2);
  AddOutput('updated_conv_state', NewConv, OnnxFloat, [1, ConvWidth, FConfig.ConvKernel]);
  ConvWeight := Reshape(Parameter(Prefix + 'conv1d.weight', [ConvWidth, 1, FConfig.ConvKernel]),
    [1, ConvWidth, FConfig.ConvKernel]);
  Mixed := Op('Mul', [NewConv, ConvWeight]);
  Mixed := Op('Mul', [Mean(Mixed, -1, False), Float(FConfig.ConvKernel)]);
  Mixed := Silu(Mixed);
  Query := Reshape(Part(Mixed, 1, 0, QKWidth), [1, FConfig.LinearKVHeads, FConfig.LinearKeySize]);
  Key := Reshape(Part(Mixed, 1, QKWidth, 2 * QKWidth),
    [1, FConfig.LinearKVHeads, FConfig.LinearKeySize]);
  Value := Reshape(Part(Mixed, 1, 2 * QKWidth, ConvWidth),
    [1, FConfig.LinearHeads, FConfig.LinearValueSize]);
  NegativeA := Parameter(Prefix + 'A_log', [FConfig.LinearHeads]);
  if FConvention = qwOriginal then NegativeA := Op('Neg', [Op('Exp', [NegativeA])]);
  Decay := Op('Mul', [NegativeA, Softplus(Op('Add', [A,
    Parameter(Prefix + 'dt_bias', [FConfig.LinearHeads])]))]);
  Decay := Reshape(Decay, [1, FConfig.LinearHeads]);
  B := Reshape(Op('Sigmoid', [B]), [1, FConfig.LinearHeads]);
  State := Input('recurrent_state',
    [1, FConfig.LinearHeads, FConfig.LinearKeySize, FConfig.LinearValueSize]);
  Output := GatedDeltaNet(Query, Key, Value, Decay, B, State,
    FConfig.LinearHeads, FConfig.LinearKVHeads, FConfig.LinearKeySize,
    FConfig.LinearValueSize, 1e-6, Updated, FConvention = qwGGUF);
  AddOutput('updated_recurrent_state', Updated, OnnxFloat,
    [1, FConfig.LinearHeads, FConfig.LinearKeySize, FConfig.LinearValueSize]);
  Output := Normalize(Output, Prefix + 'norm', FConfig.LinearValueSize, False);
  Z := Reshape(Z, [1, FConfig.LinearHeads, FConfig.LinearValueSize]);
  Output := Reshape(Op('Mul', [Output, Silu(Z)]), [1, 1, VWidth]);
  Result := Linear(Output, Prefix + 'out_proj', VWidth, FConfig.Hidden);
end;

function TQwen35LayerBuilder.PartialRotary(const X: string;
  const Factors: TRotaryEmbedding): string;
var
  Rotated: string;
begin
  Rotated := ApplyRotary(Part(X, 3, 0, FConfig.RotaryDimensions), Factors,
    FConfig.RotaryDimensions);
  if FConfig.RotaryDimensions = FConfig.HeadSize then Exit(Rotated);
  Result := AxisOp('Concat', [Rotated, Part(X, 3, FConfig.RotaryDimensions, FConfig.HeadSize)], 3);
end;

function TQwen35LayerBuilder.FullAttention(const X: string): string;
const
  Prefix = 'self_attn.';
var
  QWidth, KVWidth, I: Integer;
  QGate, Query, Key, Value, Gate, CachePosition, RopePosition, CacheMask, Valid, Past, Scores: string;
  CacheKey, CacheValue, Output: string;
  Positions: TArray<Int64>;
  Factors: TRotaryEmbedding;
begin
  QWidth := FConfig.Heads * FConfig.HeadSize;
  KVWidth := FConfig.KVHeads * FConfig.HeadSize;
  { cache_position addresses the KV cache and the causal mask. rope_position
    holds temporal, height and width positions; the selector picks one axis
    per frequency on int64 values before the FP32 cast. }
  CachePosition := Input('cache_position', [1], OnnxInt64);
  RopePosition := Input('rope_position', [3], OnnxInt64);
  Factors := RotaryPerFrequency(Gather(RopePosition, Integers(Qwen35MRopeSelector(FConfig)), 0),
    FConfig.RopeFrequency, 1, FConfig.RotaryDimensions);
  QGate := Reshape(Linear(X, Prefix + 'q_proj', FConfig.Hidden, 2 * QWidth),
    [1, 1, FConfig.Heads, 2 * FConfig.HeadSize]);
  Query := Part(QGate, 3, 0, FConfig.HeadSize);
  Gate := Reshape(Part(QGate, 3, FConfig.HeadSize, 2 * FConfig.HeadSize), [1, 1, QWidth]);
  Key := Reshape(Linear(X, Prefix + 'k_proj', FConfig.Hidden, KVWidth),
    [1, 1, FConfig.KVHeads, FConfig.HeadSize]);
  Value := Transpose(Reshape(Linear(X, Prefix + 'v_proj', FConfig.Hidden, KVWidth),
    [1, 1, FConfig.KVHeads, FConfig.HeadSize]), [0, 2, 1, 3]);
  Query := PartialRotary(Normalize(Query, Prefix + 'q_norm', FConfig.HeadSize), Factors);
  Key := PartialRotary(Normalize(Key, Prefix + 'k_norm', FConfig.HeadSize), Factors);
  Query := Transpose(Query, [0, 2, 1, 3]);
  Key := Transpose(Key, [0, 2, 1, 3]);
  CacheKey := Input('key_cache', [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  CacheValue := Input('value_cache', [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  SetLength(Positions, FContext);
  for I := 0 to FContext - 1 do Positions[I] := I;
  Past := Integers(Positions);
  CacheMask := Reshape(Op('Equal', [Past, CachePosition]), [1, 1, FContext, 1]);
  Key := Op('Where', [CacheMask, Key, CacheKey]);
  Value := Op('Where', [CacheMask, Value, CacheValue]);
  AddOutput('updated_key_cache', Key, OnnxFloat, [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  AddOutput('updated_value_cache', Value, OnnxFloat,
    [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  Key := RepeatHeads(Key, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize);
  Value := RepeatHeads(Value, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize);
  Scores := Op('MatMul', [Query, Transpose(Key, [0, 1, 3, 2])]);
  Scores := Op('Mul', [Scores, Float(1 / Sqrt(FConfig.HeadSize))]);
  Valid := Op('GreaterOrEqual', [CachePosition, Past]);
  Scores := Op('Where', [Valid, Scores, Float(-3.402823466e38)]);
  Scores := AxisOp('Softmax', [Scores], -1);
  Output := Transpose(Op('MatMul', [Scores, Value]), [0, 2, 1, 3]);
  Output := Op('Mul', [Reshape(Output, [1, 1, QWidth]), Op('Sigmoid', [Gate])]);
  Result := Linear(Output, Prefix + 'o_proj', QWidth, FConfig.Hidden);
end;

function TQwen35LayerBuilder.Build(const Config: TQwen35Config; const Kind: TQwen35Attention;
  const Context: Integer; const Convention: TQwen35WeightConvention;
  const Storage: TQwen35WeightStorage): TOnnxModel;
var
  X, Hidden, Residual, Gate, Up, Output: string;
  HiddenOutput: TOnnxValue;
begin
  RequireCpu((Config.Hidden > 0) and (Config.Intermediate > 0) and (Config.Heads > 0) and
    (Config.KVHeads > 0) and (Config.HeadSize > 0) and (Config.LinearHeads > 0) and
    (Config.LinearKVHeads > 0) and (Config.LinearKeySize > 0) and
    (Config.LinearValueSize > 0) and (Config.ConvKernel > 0) and
    (Context >= 1) and (Context <= Config.Context) and (Config.RMSEpsilon > 0),
    'Invalid Qwen layer graph dimensions');
  RequireCpu((Config.Heads mod Config.KVHeads = 0) and
    (Config.LinearHeads mod Config.LinearKVHeads = 0) and
    (Config.RotaryDimensions > 0) and (Config.RotaryDimensions <= Config.HeadSize) and
    (Config.RotaryDimensions mod 2 = 0), 'Invalid Qwen layer head or rotary dimensions');
  FConfig := Config;
  FContext := Context;
  FConvention := Convention;
  RequireCpu((Storage <> qwsBF16) or (Convention = qwOriginal),
    'BF16 graph storage requires original parameter conventions');
  FStorage := Storage;
  X := Input('hidden', [1, 1, Config.Hidden]);
  Hidden := Normalize(X, 'input_layernorm', Config.Hidden);
  if Kind = qaGatedDeltaNet then
    Hidden := LinearAttention(Hidden)
  else
    Hidden := FullAttention(Hidden);
  Residual := Op('Add', [X, Hidden]);
  Hidden := Normalize(Residual, 'post_attention_layernorm', Config.Hidden);
  Gate := Silu(Linear(Hidden, 'mlp.gate_proj', Config.Hidden, Config.Intermediate));
  Up := Linear(Hidden, 'mlp.up_proj', Config.Hidden, Config.Intermediate);
  Output := Linear(Op('Mul', [Gate, Up]), 'mlp.down_proj', Config.Intermediate, Config.Hidden);
  AddOutput('updated_hidden', Op('Add', [Residual, Output]), OnnxFloat, [1, 1, Config.Hidden]);
  HiddenOutput := FOutputs.Last;
  FOutputs.Delete(FOutputs.Count - 1);
  FOutputs.Insert(0, HiddenOutput);
  Result := CreateModel;
end;

function TQwen35LayerBuilder.BuildHead(const Config: TQwen35Config; const Rows: Integer;
  const Convention: TQwen35WeightConvention; const Storage: TQwen35WeightStorage): TOnnxModel;
var
  Hidden, Logits: string;
begin
  RequireCpu((Config.Hidden > 0) and (Rows > 0) and (Rows <= Config.Vocab) and
    (Config.RMSEpsilon > 0), 'Invalid Qwen output projection dimensions');
  RequireCpu((Storage <> qwsBF16) or (Convention = qwOriginal),
    'BF16 graph storage requires original parameter conventions');
  FConfig := Config;
  FConvention := Convention;
  FStorage := Storage;
  Hidden := Normalize(Input('hidden', [1, 1, Config.Hidden]), 'norm', Config.Hidden);
  Logits := Linear(Hidden, 'lm_head', Config.Hidden, Rows);
  AddOutput('logits', Reshape(Logits, [1, Rows]), OnnxFloat, [1, Rows]);
  Result := CreateModel;
end;

function BuildQwen35LayerGraph(const Config: TQwen35Config; const Kind: TQwen35Attention;
  const Context: Integer; const Convention: TQwen35WeightConvention;
  const Storage: TQwen35WeightStorage): TOnnxModel;
var
  Builder: TQwen35LayerBuilder;
begin
  Builder := TQwen35LayerBuilder.Create;
  try
    Result := Builder.Build(Config, Kind, Context, Convention, Storage);
  finally
    Builder.Free;
  end;
end;

function BuildQwen35HeadGraph(const Config: TQwen35Config; const Rows: Integer;
  const Convention: TQwen35WeightConvention; const Storage: TQwen35WeightStorage): TOnnxModel;
var
  Builder: TQwen35LayerBuilder;
begin
  Builder := TQwen35LayerBuilder.Create;
  try
    Result := Builder.BuildHead(Config, Rows, Convention, Storage);
  finally
    Builder.Free;
  end;
end;

end.
