{ ************************************************************************** }
{                                                                            }
{ CrossPascal.GatedDeltaNetGraph                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.GatedDeltaNetGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.ModelGraph, CrossPascal.Onnx;

type
  TGatedDeltaNetGraphBuilder = class(TModelGraphBuilder)
  private
    function Sum(const Input: string; const Axis: Integer): string;
    function L2Normalize(const Input: string; const Epsilon: Single): string;
    function GroupHeads(const Input: string; const Heads, KeyHeads, Width: Integer;
      const Tiled: Boolean): string;
  protected
    function GatedDeltaNet(const Query, Key, Value, Decay, Beta, State: string;
      const Heads, KeyHeads, KeyWidth, ValueWidth: Integer; const Epsilon: Single;
      out UpdatedState: string; const TiledHeads: Boolean = False): string;
  end;

function BuildGatedDeltaNetStep(const Heads, KeyHeads, KeyWidth, ValueWidth: Integer;
  const Epsilon: Single = 1e-6): TOnnxModel;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.CpuGraph;

type
  TGatedDeltaNetStepBuilder = class(TGatedDeltaNetGraphBuilder)
  private
    function Input(const Name: string; const Shape: array of Int64): string;
  public
    function Build(const Heads, KeyHeads, KeyWidth, ValueWidth: Integer;
      const Epsilon: Single): TOnnxModel;
  end;

function TGatedDeltaNetGraphBuilder.Sum(const Input: string; const Axis: Integer): string;
var
  Node: TOnnxNode;
begin
  Result := Op('ReduceSum', [Input, Integers([Axis])]);
  Node := FNodes.Last;
  SetLength(Node.Attributes, 1);
  Node.Attributes[0].Name := 'keepdims';
  Node.Attributes[0].IntValue := 1;
  FNodes[FNodes.Count - 1] := Node;
end;

function TGatedDeltaNetGraphBuilder.L2Normalize(const Input: string; const Epsilon: Single): string;
var
  Squares, Inverse: string;
begin
  Squares := Sum(Op('Mul', [Input, Input]), -1);
  Inverse := Op('Sqrt', [Op('Add', [Squares, Float(Epsilon)])]);
  Result := Op('Div', [Input, Inverse]);
end;

function TGatedDeltaNetGraphBuilder.GroupHeads(const Input: string;
  const Heads, KeyHeads, Width: Integer; const Tiled: Boolean): string;
begin
  if Heads = KeyHeads then Exit(Input);
  if Tiled then Exit(Op('Tile', [Input, Integers([1, Heads div KeyHeads, 1])]));
  Result := Reshape(Input, [1, KeyHeads, 1, Width]);
  Result := Op('Expand', [Result, Integers([1, KeyHeads, Heads div KeyHeads, Width])]);
  Result := Reshape(Result, [1, Heads, Width]);
end;

function TGatedDeltaNetGraphBuilder.GatedDeltaNet(const Query, Key, Value, Decay, Beta, State: string;
  const Heads, KeyHeads, KeyWidth, ValueWidth: Integer; const Epsilon: Single; out UpdatedState: string;
  const TiledHeads: Boolean): string;
var
  Q, K, KColumn, Previous, Prediction, Delta: string;
begin
  RequireCpu((Heads > 0) and (KeyHeads > 0) and (Heads mod KeyHeads = 0) and
    (KeyWidth > 0) and (ValueWidth > 0) and (Epsilon > 0) and
    not IsNan(Epsilon) and not IsInfinite(Epsilon), 'Invalid Gated DeltaNet dimensions or epsilon');
  Q := GroupHeads(L2Normalize(Query, Epsilon), Heads, KeyHeads, KeyWidth, TiledHeads);
  Q := Op('Mul', [Q, Float(1 / Sqrt(KeyWidth))]);
  K := GroupHeads(L2Normalize(Key, Epsilon), Heads, KeyHeads, KeyWidth, TiledHeads);
  KColumn := Reshape(K, [1, Heads, KeyWidth, 1]);
  Previous := Op('Mul', [State, Reshape(Op('Exp', [Decay]), [1, Heads, 1, 1])]);
  Prediction := Sum(Op('Mul', [Previous, KColumn]), -2);
  Prediction := Reshape(Prediction, [1, Heads, ValueWidth]);
  Delta := Op('Mul', [Op('Sub', [Value, Prediction]), Reshape(Beta, [1, Heads, 1])]);
  UpdatedState := Op('Add', [Previous,
    Op('Mul', [KColumn, Reshape(Delta, [1, Heads, 1, ValueWidth])])]);
  Result := Sum(Op('Mul', [UpdatedState, Reshape(Q, [1, Heads, KeyWidth, 1])]), -2);
  Result := Reshape(Result, [1, Heads, ValueWidth]);
end;

function TGatedDeltaNetStepBuilder.Input(const Name: string; const Shape: array of Int64): string;
begin
  FInputs.Add(ValueInfo(Name, OnnxFloat, Shape));
  Result := Name;
end;

function TGatedDeltaNetStepBuilder.Build(const Heads, KeyHeads, KeyWidth, ValueWidth: Integer;
  const Epsilon: Single): TOnnxModel;
var
  Q, K, V, Decay, Beta, State, Updated, Output: string;
begin
  RequireCpu((Heads >= 1) and (Heads <= 1024) and (KeyHeads >= 1) and
    (KeyHeads <= Heads) and (KeyWidth >= 1) and (KeyWidth <= 65536) and
    (ValueWidth >= 1) and (ValueWidth <= 65536), 'Gated DeltaNet dimensions are out of range');
  Q := Input('query', [1, KeyHeads, KeyWidth]);
  K := Input('key', [1, KeyHeads, KeyWidth]);
  V := Input('value', [1, Heads, ValueWidth]);
  Decay := Input('decay', [1, Heads]);
  Beta := Input('beta', [1, Heads]);
  State := Input('state', [1, Heads, KeyWidth, ValueWidth]);
  Output := GatedDeltaNet(Q, K, V, Decay, Beta, State,
    Heads, KeyHeads, KeyWidth, ValueWidth, Epsilon, Updated);
  AddOutput('output', Output, OnnxFloat, [1, Heads, ValueWidth]);
  AddOutput('updated_state', Updated, OnnxFloat, [1, Heads, KeyWidth, ValueWidth]);
  Result := CreateModel;
end;

function BuildGatedDeltaNetStep(const Heads, KeyHeads, KeyWidth, ValueWidth: Integer;
  const Epsilon: Single): TOnnxModel;
var
  Builder: TGatedDeltaNetStepBuilder;
begin
  Builder := TGatedDeltaNetStepBuilder.Create;
  try
    Result := Builder.Build(Heads, KeyHeads, KeyWidth, ValueWidth, Epsilon);
  finally
    Builder.Free;
  end;
end;

end.
