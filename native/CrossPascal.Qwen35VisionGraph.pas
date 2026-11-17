{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35VisionGraph                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35VisionGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Onnx;

{ Static single-image graph of Transformers Qwen3_5VisionModel.forward followed by
  Qwen3_5VisionPatchMerger (use_postshuffle_norm = False). ImageHeight and ImageWidth
  are resized pixel sizes, multiples of patch_size * spatial_merge_size.
  Input "patches" [N, C * T * P * P] FP32 holds HF flattened pixel_values with rows in
  blockRow, blockCol, intraRow, intraCol order and temporal grid 1. Parameters are FP32
  graph inputs named model.visual.* with source shapes. Output "image_embeddings"
  [N / merge^2, out_hidden_size] FP32. }
function BuildQwen35VisionGraph(const Root: TJSONObject; const ImageHeight, ImageWidth: Integer): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF},
  CrossPascal.ModelGraph, TVM.CpuGraph;

const
  Prefix = 'model.visual.';
  { nn.LayerNorm(eps=1e-6) in Qwen3_5VisionBlock and Qwen3_5VisionPatchMerger. }
  LayerNormEpsilon: Single = 1e-6;
  { Qwen3_5VisionRotaryEmbedding default theta; not a config field. }
  VisionRopeTheta: Double = 10000.0;
  MaxTensorElements = Int64(High(Integer));
  MaxDepth = 1 shl 16;
  OnnxAttributeFloat = 1;
  OnnxAttributeInt = 2;

type
  TQwen35VisionConfig = record
    Depth: Integer;
    Hidden: Integer;
    Heads: Integer;
    HeadSize: Integer;
    Intermediate: Integer;
    Channels: Integer;
    Temporal: Integer;
    Patch: Integer;
    Merge: Integer;
    Positions: Integer;
    Side: Integer;
    OutHidden: Integer;
    GridH: Integer;
    GridW: Integer;
    Tokens: Integer;
  end;

  TQwen35VisionBuilder = class(TModelGraphBuilder)
  private
    FConfig: TQwen35VisionConfig;
    FRow: TArray<Integer>;
    FColumn: TArray<Integer>;
    function Input(const Name: string; const Shape: array of Int64): string;
    function Linear(const X, Name: string; const Columns, Rows: Integer): string;
    function LayerNorm(const X, Name: string): string;
    function ExactGELU(const X: string): string;
    procedure TokenCoordinates;
    function PositionEmbedding: string;
    function VisionRotary: TRotaryEmbedding;
    function Attention(const X, Name: string; const Factors: TRotaryEmbedding): string;
  public
    function Build(const Config: TQwen35VisionConfig): TOnnxModel;
  end;

{ Overflow-safe product; every factor positive, total at most MaxTensorElements. }
function Elements(const Values: array of Int64): Int64;
var
  I: Integer;
begin
  Result := 1;
  for I := 0 to High(Values) do
  begin
    RequireCpu((Values[I] > 0) and (Values[I] <= MaxTensorElements div Result),
      'Qwen vision graph tensor dimensions are invalid or too large');
    Result := Result * Values[I];
  end;
end;

function Member(const Obj: TJSONObject; const Name: string): TJSONValue;
begin
  Result := Obj.GetValue(Name);
  RequireCpu(Result <> nil, 'Qwen config lacks field: ' + Name);
end;

function ConfigObject(const Obj: TJSONObject; const Name: string): TJSONObject;
var
  Value: TJSONValue;
begin
  Value := Member(Obj, Name);
  RequireCpu(Value is TJSONObject, 'Qwen config field is not an object: ' + Name);
  Result := TJSONObject(Value);
end;

function ConfigInteger(const Obj: TJSONObject; const Name: string): Int64;
var
  Value: TJSONValue;
begin
  Result := 0;
  Value := Member(Obj, Name);
  RequireCpu((Value is TJSONNumber) and TryStrToInt64(JSONValueText(Value), Result),
    'Qwen config field is not an integer: ' + Name);
end;

function ConfigText(const Obj: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := Member(Obj, Name);
  RequireCpu(Value is TJSONString, 'Qwen config field is not a string: ' + Name);
  Result := JSONValueText(Value);
end;

function ReadVisionConfig(const Root: TJSONObject;
  const ImageHeight, ImageWidth: Integer): TQwen35VisionConfig;
var
  Vision, Text: TJSONObject;
  Deepstack: TJSONValue;
  Depth, Hidden, Heads, Intermediate, Channels, Temporal, Patch, Merge: Int64;
  Positions, OutHidden, Side, Stride, GridH, GridW, Tokens, Width, Merged: Int64;
  Root64: Double;
begin
  RequireCpu(Root <> nil, 'Qwen config is missing');
  Vision := ConfigObject(Root, 'vision_config');
  Text := ConfigObject(Root, 'text_config');
  Depth := ConfigInteger(Vision, 'depth');
  Hidden := ConfigInteger(Vision, 'hidden_size');
  Heads := ConfigInteger(Vision, 'num_heads');
  Intermediate := ConfigInteger(Vision, 'intermediate_size');
  Channels := ConfigInteger(Vision, 'in_channels');
  Temporal := ConfigInteger(Vision, 'temporal_patch_size');
  Patch := ConfigInteger(Vision, 'patch_size');
  Merge := ConfigInteger(Vision, 'spatial_merge_size');
  Positions := ConfigInteger(Vision, 'num_position_embeddings');
  OutHidden := ConfigInteger(Vision, 'out_hidden_size');
  RequireCpu(ConfigText(Vision, 'hidden_act') = 'gelu_pytorch_tanh',
    'Unsupported Qwen vision block activation');
  Deepstack := Vision.GetValue('deepstack_visual_indexes');
  RequireCpu((Deepstack = nil) or ((Deepstack is TJSONArray) and (TJSONArray(Deepstack).Count = 0)),
    'Qwen vision Deepstack outputs are not supported by this graph');
  RequireCpu((Depth > 0) and (Depth <= MaxDepth), 'Invalid Qwen vision depth');
  Elements([Hidden]);
  Elements([Heads]);
  Elements([Intermediate]);
  Elements([Channels]);
  Elements([Temporal]);
  Elements([Patch]);
  Elements([Merge]);
  Elements([Positions, OutHidden]);
  RequireCpu((Hidden mod Heads = 0) and ((Hidden div Heads) mod 4 = 0),
    'Qwen vision head size must be a multiple of four for 2D rotary embedding');
  RequireCpu(OutHidden = ConfigInteger(Text, 'hidden_size'),
    'Qwen vision out_hidden_size differs from text hidden_size');
  Stride := Elements([Patch, Merge]);
  RequireCpu((ImageHeight > 0) and (ImageWidth > 0) and (ImageHeight mod Stride = 0) and (ImageWidth mod Stride = 0),
    'Qwen image size must be a positive multiple of patch_size * spatial_merge_size');
  GridH := ImageHeight div Patch;
  GridW := ImageWidth div Patch;
  Tokens := Elements([GridH, GridW]);
  Width := Elements([Channels, Temporal, Patch, Patch]);
  Merged := Elements([Hidden, Merge, Merge]);
  Elements([Tokens, Width]);
  Elements([Hidden, Width]);
  Elements([Tokens, 3, Hidden]);
  Elements([Hidden, 3, Hidden]);
  Elements([Tokens, Intermediate]);
  Elements([Intermediate, Hidden]);
  Elements([Heads, Tokens, Tokens]);
  Elements([Tokens, Hidden, SizeOf(Single)]);
  Elements([Tokens, SizeOf(Int64)]);
  Elements([Positions, Hidden]);
  Elements([Merged, Merged]);
  Elements([OutHidden, Merged]);
  { int(num_position_embeddings ** 0.5) }
  Root64 := Positions;
  Side := Trunc(Sqrt(Root64));
  while Side * Side > Positions do Dec(Side);
  while (Side + 1) * (Side + 1) <= Positions do Inc(Side);
  RequireCpu(Side >= 1, 'Invalid Qwen vision position table');
  Result.Depth := Integer(Depth);
  Result.Hidden := Integer(Hidden);
  Result.Heads := Integer(Heads);
  Result.HeadSize := Integer(Hidden div Heads);
  Result.Intermediate := Integer(Intermediate);
  Result.Channels := Integer(Channels);
  Result.Temporal := Integer(Temporal);
  Result.Patch := Integer(Patch);
  Result.Merge := Integer(Merge);
  Result.Positions := Integer(Positions);
  Result.Side := Integer(Side);
  Result.OutHidden := Integer(OutHidden);
  Result.GridH := Integer(GridH);
  Result.GridW := Integer(GridW);
  Result.Tokens := Integer(Tokens);
end;

{ torch.linspace(0, Side - 1, Grid) in FP32 as in the ATen scalar kernel (first half
  start + step * i, second half end - step * (steps - 1 - i)), then .int() truncation,
  ceil clip and the bilinear fractions of fast_pos_embed_interpolate. One operation per
  Single assignment keeps FP32 rounding. }
procedure LinearSamples(const Grid, Side: Integer; out Lower, Upper: TArray<Int64>;
  out Fraction, Rest: TArray<Single>);
var
  I, Halfway: Integer;
  Last, Step, Count, Index, Product, Value, Base: Single;
begin
  SetLength(Lower, Grid);
  SetLength(Upper, Grid);
  SetLength(Fraction, Grid);
  SetLength(Rest, Grid);
  Last := Side - 1;
  Step := 0;
  if Grid > 1 then
  begin
    Count := Grid - 1;
    Step := Last / Count;
  end;
  Halfway := Grid div 2;
  for I := 0 to Grid - 1 do
  begin
    if Grid = 1 then
      Value := 0
    else if I < Halfway then
    begin
      Index := I;
      Value := Step * Index;
    end
    else begin
      Index := Grid - 1 - I;
      Product := Step * Index;
      Value := Last - Product;
    end;
    Lower[I] := Trunc(Value);
    if Lower[I] + 1 < Side - 1 then
      Upper[I] := Lower[I] + 1
    else
      Upper[I] := Side - 1;
    Base := Lower[I];
    Fraction[I] := Value - Base;
    Rest[I] := 1 - Fraction[I];
  end;
end;

function TQwen35VisionBuilder.Input(const Name: string; const Shape: array of Int64): string;
begin
  FInputs.Add(ValueInfo(Name, OnnxFloat, Shape));
  Result := Name;
end;

function TQwen35VisionBuilder.Linear(const X, Name: string; const Columns, Rows: Integer): string;
begin
  Result := Op('MatMul', [X, Transpose(Input(Name + '.weight', [Rows, Columns]), [1, 0])]);
  Result := Op('Add', [Result, Input(Name + '.bias', [Rows])]);
end;

function TQwen35VisionBuilder.LayerNorm(const X, Name: string): string;
var
  Node: TOnnxNode;
begin
  Result := Op('LayerNormalization', [X, Input(Name + '.weight', [FConfig.Hidden]),
    Input(Name + '.bias', [FConfig.Hidden])]);
  Node := FNodes.Last;
  SetLength(Node.Attributes, 2);
  Node.Attributes[0].Name := 'axis';
  Node.Attributes[0].Kind := OnnxAttributeInt;
  Node.Attributes[0].IntValue := -1;
  Node.Attributes[1].Name := 'epsilon';
  Node.Attributes[1].Kind := OnnxAttributeFloat;
  Node.Attributes[1].FloatValue := LayerNormEpsilon;
  FNodes[FNodes.Count - 1] := Node;
end;

{ nn.GELU() default: x * 0.5 * (1 + erf(x * M_SQRT1_2)). }
function TQwen35VisionBuilder.ExactGELU(const X: string): string;
begin
  Result := Op('Mul', [Op('Mul', [X, Float(0.5)]),
    Op('Add', [Float(1), Op('Erf', [Op('Mul', [X, Float(Sqrt(0.5))])])])]);
end;

{ Row/column of every token in blockRow, blockCol, intraRow, intraCol order, as in
  rot_pos_emb and the permute of fast_pos_embed_interpolate. }
procedure TQwen35VisionBuilder.TokenCoordinates;
var
  BlockRow, BlockCol, IntraRow, IntraCol, T, M: Integer;
begin
  M := FConfig.Merge;
  SetLength(FRow, FConfig.Tokens);
  SetLength(FColumn, FConfig.Tokens);
  T := 0;
  for BlockRow := 0 to FConfig.GridH div M - 1 do
    for BlockCol := 0 to FConfig.GridW div M - 1 do
      for IntraRow := 0 to M - 1 do
        for IntraCol := 0 to M - 1 do
        begin
          FRow[T] := BlockRow * M + IntraRow;
          FColumn[T] := BlockCol * M + IntraCol;
          Inc(T);
        end;
end;

{ fast_pos_embed_interpolate: four Gathers of pos_embed.weight, FP32 bilinear weights,
  sum p0 + p1 + p2 + p3 in source order; indices are already in token order. }
function TQwen35VisionBuilder.PositionEmbedding: string;
var
  RowLow, RowHigh, ColLow, ColHigh: TArray<Int64>;
  RowFraction, RowRest, ColFraction, ColRest: TArray<Single>;
  Indices: array[0..3] of TArray<Int64>;
  Weights: array[0..3] of TArray<Single>;
  Table, Term: string;
  T, R, C, Corner, Side: Integer;
begin
  Side := FConfig.Side;
  LinearSamples(FConfig.GridH, Side, RowLow, RowHigh, RowFraction, RowRest);
  LinearSamples(FConfig.GridW, Side, ColLow, ColHigh, ColFraction, ColRest);
  for Corner := 0 to 3 do
  begin
    SetLength(Indices[Corner], FConfig.Tokens);
    SetLength(Weights[Corner], FConfig.Tokens);
  end;
  for T := 0 to FConfig.Tokens - 1 do
  begin
    R := FRow[T];
    C := FColumn[T];
    Indices[0][T] := RowLow[R] * Side + ColLow[C];
    Indices[1][T] := RowLow[R] * Side + ColHigh[C];
    Indices[2][T] := RowHigh[R] * Side + ColLow[C];
    Indices[3][T] := RowHigh[R] * Side + ColHigh[C];
    Weights[0][T] := RowRest[R] * ColRest[C];
    Weights[1][T] := RowRest[R] * ColFraction[C];
    Weights[2][T] := RowFraction[R] * ColRest[C];
    Weights[3][T] := RowFraction[R] * ColFraction[C];
  end;
  Table := Input(Prefix + 'pos_embed.weight', [FConfig.Positions, FConfig.Hidden]);
  Result := '';
  for Corner := 0 to 3 do
  begin
    Term := Op(
      'Mul',
      [
        Gather(
          Table,
          Integers(Indices[Corner]),
          0
        ),
        Constant(
          @Weights[Corner][0],
          FConfig.Tokens * SizeOf(Single),
          OnnxFloat,
          [
            FConfig.Tokens,
            1
          ]
        )
      ]
    );
    if Corner = 0 then
      Result := Term
    else
      Result := Op('Add', [Result, Term]);
  end;
end;

{ Qwen3_5VisionRotaryEmbedding(head_dim // 2): inv_freq = 1 / theta^(arange(0, dim, 2) / dim)
  with head_dim / 4 entries; per token [row * inv, column * inv], then concatenated with
  itself to head_dim. Phases are FP32 constants; Cos/Sin run in the graph. }
function TQwen35VisionBuilder.VisionRotary: TRotaryEmbedding;
var
  Inverse, Phase: TArray<Single>;
  RotaryDim, Count, J, T, D, Offset: Integer;
  Numerator, Denominator, Exponent, Power32, One, Row, Column, RowPhase, ColumnPhase: Single;
  Exponent64: Double;
  Angles: string;
begin
  D := FConfig.HeadSize;
  RotaryDim := D div 2;
  Count := (RotaryDim + 1) div 2;
  RequireCpu(4 * Count = D, 'Qwen vision rotary width differs from head size');
  SetLength(Inverse, Count);
  One := 1;
  Denominator := RotaryDim;
  for J := 0 to Count - 1 do
  begin
    Numerator := 2 * J;
    Exponent := Numerator / Denominator;
    Exponent64 := Exponent;
    Power32 := Power(VisionRopeTheta, Exponent64);
    Inverse[J] := One / Power32;
  end;
  SetLength(Phase, FConfig.Tokens * D);
  for T := 0 to FConfig.Tokens - 1 do
  begin
    Row := FRow[T];
    Column := FColumn[T];
    Offset := T * D;
    for J := 0 to Count - 1 do
    begin
      RowPhase := Row * Inverse[J];
      ColumnPhase := Column * Inverse[J];
      Phase[Offset + J] := RowPhase;
      Phase[Offset + Count + J] := ColumnPhase;
      Phase[Offset + 2 * Count + J] := RowPhase;
      Phase[Offset + 3 * Count + J] := ColumnPhase;
    end;
  end;
  Angles := Constant(@Phase[0], FConfig.Tokens * D * SizeOf(Single), OnnxFloat, [1, 1, FConfig.Tokens, D]);
  Result.Cosine := Op('Cos', [Angles]);
  Result.Sine := Op('Sin', [Angles]);
end;

{ Qwen3_5VisionAttention, eager path, one non-causal sequence:
  qkv.reshape(N, 3, H, D) -> [3, H, N, D]; q*cos + rotate_half(q)*sin;
  softmax(q k^T * D^-0.5) v -> [N, H * D] -> proj. }
function TQwen35VisionBuilder.Attention(const X, Name: string; const Factors: TRotaryEmbedding): string;
var
  N, H, D, Hidden: Integer;
  QKV, Query, Key, Value, Scores: string;
  Scale: Double;
begin
  N := FConfig.Tokens;
  H := FConfig.Heads;
  D := FConfig.HeadSize;
  Hidden := FConfig.Hidden;
  QKV := Linear(X, Name + '.qkv', Hidden, 3 * Hidden);
  QKV := Transpose(Reshape(QKV, [N, 3, H, D]), [1, 2, 0, 3]);
  Query := ApplyRotary(Part(QKV, 0, 0, 1), Factors, D);
  Key := ApplyRotary(Part(QKV, 0, 1, 2), Factors, D);
  Value := Part(QKV, 0, 2, 3);
  Scale := D;
  Scale := Power(Scale, -0.5);
  Scores := Op('Mul', [Op('MatMul', [Query, Transpose(Key, [0, 1, 3, 2])]), Float(Scale)]);
  Scores := AxisOp('Softmax', [Scores], -1);
  Result := Transpose(Op('MatMul', [Scores, Value]), [0, 2, 1, 3]);
  Result := Linear(Reshape(Result, [N, Hidden]), Name + '.proj', Hidden, Hidden);
end;

function TQwen35VisionBuilder.Build(const Config: TQwen35VisionConfig): TOnnxModel;
var
  Width, Merged, Groups, Block: Integer;
  State, Branch, Name: string;
  Factors: TRotaryEmbedding;
begin
  FConfig := Config;
  TokenCoordinates;
  Width := Config.Channels * Config.Temporal * Config.Patch * Config.Patch;
  { Qwen3_5VisionPatchEmbed: Conv3d with kernel = stride over one full C, T, H, W patch. }
  State := Input('patches', [Config.Tokens, Width]);
  Branch := Reshape(
    Input(
      Prefix + 'patch_embed.proj.weight',
      [
        Config.Hidden,
        Config.Channels,
        Config.Temporal,
        Config.Patch,
        Config.Patch
      ]
    ),
    [
      Config.Hidden,
      Width
    ]
  );
  State := Op('MatMul', [State, Transpose(Branch, [1, 0])]);
  State := Op('Add', [State, Input(Prefix + 'patch_embed.proj.bias', [Config.Hidden])]);
  State := Op('Add', [State, PositionEmbedding]);
  Factors := VisionRotary;
  for Block := 0 to Config.Depth - 1 do
  begin
    Name := Prefix + 'blocks.' + IntToStr(Block) + '.';
    State := Op('Add', [State, Attention(LayerNorm(State, Name + 'norm1'), Name + 'attn', Factors)]);
    Branch := Linear(LayerNorm(State, Name + 'norm2'), Name + 'mlp.linear_fc1', Config.Hidden,
      Config.Intermediate);
    Branch := Linear(GELU(Branch), Name + 'mlp.linear_fc2', Config.Intermediate, Config.Hidden);
    State := Op('Add', [State, Branch]);
  end;
  { Qwen3_5VisionPatchMerger: LayerNorm before the merge-group view. }
  Merged := Config.Hidden * Config.Merge * Config.Merge;
  Groups := Config.Tokens div (Config.Merge * Config.Merge);
  State := Reshape(LayerNorm(State, Prefix + 'merger.norm'), [Groups, Merged]);
  State := ExactGELU(Linear(State, Prefix + 'merger.linear_fc1', Merged, Merged));
  State := Linear(State, Prefix + 'merger.linear_fc2', Merged, Config.OutHidden);
  AddOutput('image_embeddings', State, OnnxFloat, [Groups, Config.OutHidden]);
  Result := CreateModel;
end;

function BuildQwen35VisionGraph(const Root: TJSONObject; const ImageHeight, ImageWidth: Integer): TOnnxModel;
var
  Config: TQwen35VisionConfig;
  Builder: TQwen35VisionBuilder;
begin
  Config := ReadVisionConfig(Root, ImageHeight, ImageWidth);
  Builder := TQwen35VisionBuilder.Create;
  try
    Result := Builder.Build(Config);
  finally
    Builder.Free;
  end;
end;

end.
