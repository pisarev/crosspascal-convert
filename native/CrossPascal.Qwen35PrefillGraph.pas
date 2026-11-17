{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35PrefillGraph                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35PrefillGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Onnx, CrossPascal.Qwen35Source, CrossPascal.Qwen35Graph;

{ One decoder layer over Chunk consecutive rows. Weight projections, norms,
  residuals and the MLP run once for the whole chunk; the KV cache writes,
  attention and the Gated DeltaNet recurrence are unrolled row by row with the
  same operators as BuildQwen35LayerGraph. State inputs/outputs and parameter
  names, shapes and types equal the single-step graph. }
function BuildQwen35PrefillGraph(const Config: TQwen35Config; const Kind: TQwen35Attention;
  const Context, Chunk: Integer; const Convention: TQwen35WeightConvention;
  const Storage: TQwen35WeightStorage = qwsFloat32): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Generics.Collections{$ELSE}System.SysUtils, System.Math,
  System.Generics.Collections{$ENDIF}, CrossPascal.ModelGraph,
  CrossPascal.GatedDeltaNetGraph, TVM.CpuGraph;

const
  MaxPrefillChunk = 64;

type
  TQwen35PrefillBuilder = class(TGatedDeltaNetGraphBuilder)
  private
    FConfig: TQwen35Config;
    FContext: Integer;
    FChunk: Integer;
    FConvention: TQwen35WeightConvention;
    FStorage: TQwen35WeightStorage;
    FShapes: TDictionary<string, string>;
    FFloats: TDictionary<string, string>;
    function Input(const Name: string; const Shape: array of Int64;
      const DataType: Integer = OnnxFloat): string;
    function Parameter(const Name: string; const Shape: array of Int64;
      const KeepBF16: Boolean = False): string;
    function Linear(const X, Name: string; const Columns, Rows: Integer): string;
    function Normalize(const X, Name: string; const Width: Integer;
      const ZeroCentered: Boolean = True): string;
    function Silu(const X: string): string;
    function Softplus(const X: string): string;
    function Join(const Items: array of string; const Axis: Integer): string;
    function PartialRotary(const X: string; const Factors: TRotaryEmbedding): string;
    function LinearAttention(const X: string): string;
    function FullAttention(const X: string): string;
  public
    constructor Create;
    destructor Destroy; override;
    function Build(const Config: TQwen35Config; const Kind: TQwen35Attention;
      const Context, Chunk: Integer; const Convention: TQwen35WeightConvention;
      const Storage: TQwen35WeightStorage): TOnnxModel;
  end;

{ Copy of the private CrossPascal.Qwen35Graph selector; keep both identical. }
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

function ShapeKey(const Shape: array of Int64): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(Shape) do Result := Result + IntToStr(Shape[I]) + ',';
end;

constructor TQwen35PrefillBuilder.Create;
begin
  inherited Create;
  FShapes := TDictionary<string, string>.Create;
  FFloats := TDictionary<string, string>.Create;
end;

destructor TQwen35PrefillBuilder.Destroy;
begin
  FFloats.Free;
  FShapes.Free;
  inherited;
end;

function TQwen35PrefillBuilder.Input(const Name: string; const Shape: array of Int64;
  const DataType: Integer): string;
begin
  FInputs.Add(ValueInfo(Name, DataType, Shape));
  Result := Name;
end;

{ Every weight is declared once. The raw input and its FP32 cast are cached
  separately so BF16MatMul keeps the BF16 tensor while small parameters share
  one Cast node. }
function TQwen35PrefillBuilder.Parameter(const Name: string; const Shape: array of Int64;
  const KeepBF16: Boolean): string;
var
  Known: string;
begin
  if FShapes.TryGetValue(Name, Known) then
    RequireCpu(Known = ShapeKey(Shape), 'Qwen prefill parameter has two shapes: ' + Name)
  else begin
    FShapes.Add(Name, ShapeKey(Shape));
    if FStorage = qwsBF16 then
      Input(Name, Shape, OnnxBFloat16)
    else
      Input(Name, Shape);
  end;
  Result := Name;
  if (FStorage <> qwsBF16) or KeepBF16 then Exit;
  if not FFloats.TryGetValue(Name, Result) then
  begin
    Result := Cast(Name, OnnxFloat);
    FFloats.Add(Name, Result);
  end;
end;

{ X is [1, Chunk, Columns]; all three kernels iterate over every outer axis. }
function TQwen35PrefillBuilder.Linear(const X, Name: string; const Columns, Rows: Integer): string;
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

function TQwen35PrefillBuilder.Normalize(const X, Name: string; const Width: Integer;
  const ZeroCentered: Boolean): string;
var
  W: string;
begin
  W := Parameter(Name + '.weight', [Width]);
  if ZeroCentered and (FConvention = qwOriginal) then W := Op('Add', [W, Float(1)]);
  Result := Op('Mul', [RMSNormalize(X, FConfig.RMSEpsilon), W]);
end;

function TQwen35PrefillBuilder.Silu(const X: string): string;
begin
  Result := Op('Mul', [X, Op('Sigmoid', [X])]);
end;

function TQwen35PrefillBuilder.Softplus(const X: string): string;
var
  Absolute: string;
begin
  Absolute := Op('Max', [X, Op('Neg', [X])]);
  Result := Op('Add', [Op('Max', [X, Float(0)]),
    Op('Log', [Op('Add', [Float(1), Op('Exp', [Op('Neg', [Absolute])])])])]);
end;

function TQwen35PrefillBuilder.Join(const Items: array of string; const Axis: Integer): string;
begin
  if Length(Items) = 1 then Exit(Items[0]);
  Result := AxisOp('Concat', Items, Axis);
end;

function TQwen35PrefillBuilder.PartialRotary(const X: string;
  const Factors: TRotaryEmbedding): string;
var
  Rotated: string;
begin
  Rotated := ApplyRotary(Part(X, 3, 0, FConfig.RotaryDimensions), Factors,
    FConfig.RotaryDimensions);
  if FConfig.RotaryDimensions = FConfig.HeadSize then Exit(Rotated);
  Result := AxisOp('Concat', [Rotated, Part(X, 3, FConfig.RotaryDimensions, FConfig.HeadSize)], 3);
end;

function TQwen35PrefillBuilder.LinearAttention(const X: string): string;
const
  Prefix = 'linear_attn.';
var
  QKWidth, VWidth, ConvWidth, Kernel, T: Integer;
  Mixed, Window, Conv, ConvWeight, Z, A, B, Decay, NegativeA, Row: string;
  State, Updated, Output: string;
  Windows, Outputs: TArray<string>;
begin
  QKWidth := FConfig.LinearKVHeads * FConfig.LinearKeySize;
  VWidth := FConfig.LinearHeads * FConfig.LinearValueSize;
  ConvWidth := QKWidth * 2 + VWidth;
  Kernel := FConfig.ConvKernel;
  { Batched projections: [1, Chunk, Width]. }
  Mixed := Linear(X, Prefix + 'in_proj_qkv', FConfig.Hidden, ConvWidth);
  Z := Linear(X, Prefix + 'in_proj_z', FConfig.Hidden, VWidth);
  A := Linear(X, Prefix + 'in_proj_a', FConfig.Hidden, FConfig.LinearHeads);
  B := Linear(X, Prefix + 'in_proj_b', FConfig.Hidden, FConfig.LinearHeads);
  { Sequential shifts of conv_state equal sliding windows over
    [conv_state[:, :, 1:], x_0 .. x_(Chunk-1)]: row t sees columns t .. t+Kernel-1,
    the last window is the final state. Pure data movement, no arithmetic. }
  Conv := Input('conv_state', [1, ConvWidth, Kernel]);
  Window := Transpose(Mixed, [0, 2, 1]);
  if Kernel > 1 then
    Window := AxisOp('Concat', [Part(Conv, 2, 1, Kernel), Window], 2);
  SetLength(Windows, FChunk);
  for T := 0 to FChunk - 1 do Windows[T] := Part(Window, 2, T, T + Kernel);
  AddOutput('updated_conv_state', Windows[FChunk - 1], OnnxFloat, [1, ConvWidth, Kernel]);
  ConvWeight := Reshape(Parameter(Prefix + 'conv1d.weight', [ConvWidth, 1, Kernel]),
    [1, ConvWidth, Kernel]);
  Mixed := Op('Mul', [Join(Windows, 0), ConvWeight]);
  Mixed := Op('Mul', [Mean(Mixed, -1, False), Float(Kernel)]);
  Mixed := Silu(Mixed);
  NegativeA := Parameter(Prefix + 'A_log', [FConfig.LinearHeads]);
  if FConvention = qwOriginal then NegativeA := Op('Neg', [Op('Exp', [NegativeA])]);
  Decay := Op('Mul', [NegativeA, Softplus(Op('Add', [A,
    Parameter(Prefix + 'dt_bias', [FConfig.LinearHeads])]))]);
  Decay := Reshape(Decay, [FChunk, FConfig.LinearHeads]);
  B := Reshape(Op('Sigmoid', [B]), [FChunk, FConfig.LinearHeads]);
  State := Input('recurrent_state',
    [1, FConfig.LinearHeads, FConfig.LinearKeySize, FConfig.LinearValueSize]);
  SetLength(Outputs, FChunk);
  for T := 0 to FChunk - 1 do
  begin
    Row := Part(Mixed, 0, T, T + 1);
    Outputs[T] := GatedDeltaNet(
      Reshape(
        Part(
          Row,
          1,
          0,
          QKWidth
        ),
        [
          1,
          FConfig.LinearKVHeads,
          FConfig.LinearKeySize
        ]
      ),
      Reshape(
        Part(
          Row,
          1,
          QKWidth,
          2 * QKWidth
        ),
        [
          1,
          FConfig.LinearKVHeads,
          FConfig.LinearKeySize
        ]
      ),
      Reshape(
        Part(
          Row,
          1,
          2 * QKWidth,
          ConvWidth
        ),
        [
          1,
          FConfig.LinearHeads,
          FConfig.LinearValueSize
        ]
      ),
      Part(
        Decay,
        0,
        T,
        T + 1
      ),
      Part(
        B,
        0,
        T,
        T + 1
      ),
      State,
      FConfig.LinearHeads,
      FConfig.LinearKVHeads,
      FConfig.LinearKeySize,
      FConfig.LinearValueSize,
      1e-6,
      Updated,
      FConvention = qwGGUF
    );
    State := Updated;
  end;
  AddOutput('updated_recurrent_state', State, OnnxFloat,
    [1, FConfig.LinearHeads, FConfig.LinearKeySize, FConfig.LinearValueSize]);
  { Row-wise norm and gate over [Chunk, Heads, ValueSize]. }
  Output := Normalize(Join(Outputs, 0), Prefix + 'norm', FConfig.LinearValueSize, False);
  Z := Reshape(Z, [FChunk, FConfig.LinearHeads, FConfig.LinearValueSize]);
  Output := Reshape(Op('Mul', [Output, Silu(Z)]), [1, FChunk, VWidth]);
  Result := Linear(Output, Prefix + 'out_proj', VWidth, FConfig.Hidden);
end;

function TQwen35PrefillBuilder.FullAttention(const X: string): string;
const
  Prefix = 'self_attn.';
var
  QWidth, KVWidth, I, T: Integer;
  QGate, QueryAll, KeyAll, ValueAll, GateAll, CachePositions, RopePositions: string;
  Position, Past, CacheMask, Valid, Scale, Lowest, Scores, Key, Value: string;
  Rows: TArray<string>;
  Positions: TArray<Int64>;
  Factors: TRotaryEmbedding;
begin
  QWidth := FConfig.Heads * FConfig.HeadSize;
  KVWidth := FConfig.KVHeads * FConfig.HeadSize;
  CachePositions := Input('cache_position', [FChunk], OnnxInt64);
  RopePositions := Input('rope_position', [FChunk, 3], OnnxInt64);
  { Same int64 selection, FP32 cast and phase per row as single-step; rows are
    independent, so factors are [1, Chunk, 1, RotaryDimensions]. }
  Factors := RotaryPerFrequency(Gather(RopePositions, Integers(Qwen35MRopeSelector(FConfig)), 1),
    FConfig.RopeFrequency, FChunk, FConfig.RotaryDimensions);
  QGate := Reshape(Linear(X, Prefix + 'q_proj', FConfig.Hidden, 2 * QWidth),
    [1, FChunk, FConfig.Heads, 2 * FConfig.HeadSize]);
  GateAll := Reshape(Part(QGate, 3, FConfig.HeadSize, 2 * FConfig.HeadSize), [1, FChunk, QWidth]);
  QueryAll := Part(QGate, 3, 0, FConfig.HeadSize);
  KeyAll := Reshape(Linear(X, Prefix + 'k_proj', FConfig.Hidden, KVWidth),
    [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]);
  ValueAll := Transpose(Reshape(Linear(X, Prefix + 'v_proj', FConfig.Hidden, KVWidth),
    [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]), [0, 2, 1, 3]);
  QueryAll := PartialRotary(Normalize(QueryAll, Prefix + 'q_norm', FConfig.HeadSize), Factors);
  KeyAll := PartialRotary(Normalize(KeyAll, Prefix + 'k_norm', FConfig.HeadSize), Factors);
  QueryAll := Transpose(QueryAll, [0, 2, 1, 3]);
  KeyAll := Transpose(KeyAll, [0, 2, 1, 3]);
  Key := Input('key_cache', [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  Value := Input('value_cache', [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  SetLength(Positions, FContext);
  for I := 0 to FContext - 1 do Positions[I] := I;
  Past := Integers(Positions);
  Scale := Float(1 / Sqrt(FConfig.HeadSize));
  Lowest := Float(-3.402823466e38);
  SetLength(Rows, FChunk);
  for T := 0 to FChunk - 1 do
  begin
    Position := Part(CachePositions, 0, T, T + 1);
    CacheMask := Reshape(Op('Equal', [Past, Position]), [1, 1, FContext, 1]);
    Key := Op('Where', [CacheMask, Part(KeyAll, 2, T, T + 1), Key]);
    Value := Op('Where', [CacheMask, Part(ValueAll, 2, T, T + 1), Value]);
    Scores := Op(
      'MatMul',
      [
        Part(
          QueryAll,
          2,
          T,
          T + 1
        ),
        Transpose(
          RepeatHeads(
            Key,
            FConfig.Heads,
            FConfig.KVHeads,
            FContext,
            FConfig.HeadSize
          ),
          [
            0,
            1,
            3,
            2
          ]
        )
      ]
    );
    Scores := Op('Mul', [Scores, Scale]);
    Valid := Op('GreaterOrEqual', [Position, Past]);
    Scores := Op('Where', [Valid, Scores, Lowest]);
    Scores := AxisOp('Softmax', [Scores], -1);
    Rows[T] := Op('MatMul', [Scores,
      RepeatHeads(Value, FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize)]);
    Rows[T] := Reshape(Transpose(Rows[T], [0, 2, 1, 3]), [1, 1, QWidth]);
  end;
  AddOutput('updated_key_cache', Key, OnnxFloat, [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  AddOutput('updated_value_cache', Value, OnnxFloat,
    [1, FConfig.KVHeads, FContext, FConfig.HeadSize]);
  Result := Op('Mul', [Join(Rows, 1), Op('Sigmoid', [GateAll])]);
  Result := Linear(Result, Prefix + 'o_proj', QWidth, FConfig.Hidden);
end;

function TQwen35PrefillBuilder.Build(const Config: TQwen35Config; const Kind: TQwen35Attention;
  const Context, Chunk: Integer; const Convention: TQwen35WeightConvention;
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
  RequireCpu((Chunk >= 1) and (Chunk <= MaxPrefillChunk) and (Chunk <= Context),
    'Invalid Qwen prefill chunk size');
  RequireCpu((Storage <> qwsBF16) or (Convention = qwOriginal),
    'BF16 graph storage requires original parameter conventions');
  FConfig := Config;
  FContext := Context;
  FChunk := Chunk;
  FConvention := Convention;
  FStorage := Storage;
  X := Input('hidden', [1, Chunk, Config.Hidden]);
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
  AddOutput('updated_hidden', Op('Add', [Residual, Output]), OnnxFloat, [1, Chunk, Config.Hidden]);
  HiddenOutput := FOutputs.Last;
  FOutputs.Delete(FOutputs.Count - 1);
  FOutputs.Insert(0, HiddenOutput);
  Result := CreateModel;
end;

function BuildQwen35PrefillGraph(const Config: TQwen35Config; const Kind: TQwen35Attention;
  const Context, Chunk: Integer; const Convention: TQwen35WeightConvention;
  const Storage: TQwen35WeightStorage): TOnnxModel;
var
  Builder: TQwen35PrefillBuilder;
begin
  Builder := TQwen35PrefillBuilder.Create;
  try
    Result := Builder.Build(Config, Kind, Context, Chunk, Convention, Storage);
  finally
    Builder.Free;
  end;
end;

end.
