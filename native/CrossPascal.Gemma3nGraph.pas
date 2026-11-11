{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nGraph                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Gemma3nSource, CrossPascal.Onnx;

function BuildGemma3nTextGraph(const Config: TGemma3nConfig; const Bindings: TArray<TGemma3nBinding>;
  const Context, Chunk: Integer; const Quantized: Boolean = False): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Generics.Collections{$ELSE}System.SysUtils, System.Math, System.Generics.Collections{$ENDIF},
  CrossPascal.ModelGraph, TVM.CpuGraph;

type
  TGemma3nGraphBuilder = class(TModelGraphBuilder)
  private
    FConfig: TGemma3nConfig;
    FContext: Integer;
    FQuantized: Boolean;
    FChunk: Integer;
    FCacheLayers: Integer;
    FWeights: TDictionary<string, TGemma3nBinding>;
    FUsedWeights: TDictionary<string, string>;
    FPosition: string;
    FCacheMask: string;
    FCacheIndices: string;
    FFullMask: string;
    FSlidingMask: string;
    FFullRotary: TRotaryEmbedding;
    FSlidingRotary: TRotaryEmbedding;
    FKeys: TArray<string>;
    FValues: TArray<string>;
    function Weight(const Name: string): string;
    function Linear(const Input, WeightName: string): string;
    function Normalize(const Input: string; const WeightName: string = ''): string;
    function Router(const Input, Prefix: string): string;
    function Attention(const Input, Prefix: string; const Layer: Integer): string;
    function Layer(const Hidden, PerLayer: string; const Index: Integer): string;
    function MatchMagnitude(const Input, Magnitude: string): string;
    procedure AddCacheOperations;
  public
    constructor Create(const Config: TGemma3nConfig; const Bindings: TArray<TGemma3nBinding>;
      const Context, Chunk: Integer; const Quantized: Boolean);
    destructor Destroy; override;
    function Build: TOnnxModel;
  end;

constructor TGemma3nGraphBuilder.Create(const Config: TGemma3nConfig;
  const Bindings: TArray<TGemma3nBinding>; const Context, Chunk: Integer; const Quantized: Boolean);
var
  Binding: TGemma3nBinding;
begin
  inherited Create;
  RequireCpu(
    (Context >= 2) and (Context <= Config.Context) and (Chunk >= 1) and (Chunk <= Context) and (Config.HeadSize mod 2 = 0) and (Config.KVHeads > 0) and (Config.Heads mod Config.KVHeads = 0),
    'Invalid Gemma 3n graph configuration'
  );
  FConfig := Config;
  FContext := Context;
  FQuantized := Quantized;
  FChunk := Chunk;
  FCacheLayers := Config.Layers - Config.SharedLayers;
  SetLength(FKeys, FCacheLayers);
  SetLength(FValues, FCacheLayers);
  FWeights := TDictionary<string, TGemma3nBinding>.Create;
  FUsedWeights := TDictionary<string, string>.Create;
  for Binding in Bindings do FWeights.Add(Binding.Name, Binding);
end;

destructor TGemma3nGraphBuilder.Destroy;
begin
  FUsedWeights.Free;
  FWeights.Free;
  inherited;
end;

function TGemma3nGraphBuilder.Weight(const Name: string): string;
var
  Binding: TGemma3nBinding;
begin
  if FUsedWeights.TryGetValue(Name, Result) then Exit;
  RequireCpu(FWeights.TryGetValue(Name, Binding), 'Missing Gemma 3n graph weight: ' + Name);
  Result := 'text.' + Name;
  FInputs.Add(ValueInfo(Result, 1, Binding.Shape));
  FUsedWeights.Add(Name, Result);
end;

function TGemma3nGraphBuilder.Linear(const Input, WeightName: string): string;
var
  Binding: TGemma3nBinding;
  Q8, Scale: string;
  Node: TOnnxNode;
begin
  if FQuantized then
  begin
    RequireCpu(FWeights.TryGetValue(WeightName, Binding), 'Missing Gemma 3n graph weight: ' + WeightName);
    if (Length(Binding.Shape) = 2) and (TensorElements(Binding.Shape) >= 256) then
    begin
      Q8 := 'text.' + WeightName + '_q8';
      Scale := 'text.' + WeightName + '_scale';
      if not FUsedWeights.ContainsKey(Q8) then
      begin
        FInputs.Add(ValueInfo(Q8, 3, Binding.Shape));
        FInputs.Add(ValueInfo(Scale, 1, [Binding.Shape[0]]));
        FUsedWeights.Add(Q8, Q8);
      end;
      Result := Op('Q8MatMul', [Input, Q8, Scale]);
      Node := FNodes.Last;
      Node.Domain := 'crosspascal';
      FNodes[FNodes.Count - 1] := Node;
      Exit;
    end;
  end;
  Result := Op('MatMul', [Input, Transpose(Weight(WeightName), [1, 0])]);
end;

function TGemma3nGraphBuilder.Normalize(const Input, WeightName: string): string;
begin
  Result := RMSNormalize(Input, FConfig.RMSEpsilon);
  if WeightName <> '' then Result := Op('Mul', [Result, Weight(WeightName)]);
end;

function TGemma3nGraphBuilder.Router(const Input, Prefix: string): string;
var
  Norm: string;
begin
  Norm := Normalize(Input, Prefix + 'router_norm.weight');
  Norm := Op('Mul', [Norm, Float(1 / FConfig.Hidden)]);
  Result := Op('Tanh', [Linear(Norm, Prefix + 'modality_router.weight')]);
end;

function TGemma3nGraphBuilder.Attention(const Input, Prefix: string; const Layer: Integer): string;
var
  Factors: TRotaryEmbedding;
  Mask, Queries, Key, Value, Scores, Context: string;
  Shared: Integer;
begin
  if FConfig.Sliding[Layer] then
  begin
    Factors := FSlidingRotary;
    Mask := FSlidingMask;
  end
  else begin
    Factors := FFullRotary;
    Mask := FFullMask;
  end;
  Queries := Reshape(Linear(Input, Prefix + 'q_proj.weight'), [1, FChunk, FConfig.Heads, FConfig.HeadSize]);
  Queries := Normalize(Queries, Prefix + 'q_norm.weight');
  Queries := Transpose(ApplyRotary(Queries, Factors, FConfig.HeadSize), [0, 2, 1, 3]);
  if Layer < FCacheLayers then
  begin
    Key := Reshape(Linear(Input, Prefix + 'k_proj.weight'), [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]);
    Key := Normalize(Key, Prefix + 'k_norm.weight');
    Key := Transpose(ApplyRotary(Key, Factors, FConfig.HeadSize), [0, 2, 1, 3]);
    Value := Reshape(Linear(Input, Prefix + 'v_proj.weight'), [1, FChunk, FConfig.KVHeads, FConfig.HeadSize]);
    Value := Transpose(Normalize(Value), [0, 2, 1, 3]);
    if FChunk > 1 then
    begin
      Key := Gather(Key, FCacheIndices, 2);
      Value := Gather(Value, FCacheIndices, 2);
    end;
    FKeys[Layer] := Op('Where', [FCacheMask, Key, 'cache_' + IntToStr(2 * Layer)]);
    FValues[Layer] := Op('Where', [FCacheMask, Value, 'cache_' + IntToStr(2 * Layer + 1)]);
    Shared := Layer;
  end
  else begin
    Shared := FCacheLayers - 1;
    while (Shared >= 0) and (FConfig.Sliding[Shared] <> FConfig.Sliding[Layer]) do Dec(Shared);
    RequireCpu(Shared >= 0, 'Missing compatible shared Gemma 3n KV cache');
  end;
  Key := RepeatHeads(FKeys[Shared], FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize);
  Key := Transpose(Key, [0, 1, 3, 2]);
  Value := RepeatHeads(FValues[Shared], FConfig.Heads, FConfig.KVHeads, FContext, FConfig.HeadSize);
  Scores := Op('MatMul', [Queries, Key]);
  Scores := AxisOp('Softmax', [Op('Add', [Scores, Mask])], -1);
  Context := Transpose(Op('MatMul', [Scores, Value]), [0, 2, 1, 3]);
  Context := Reshape(Context, [1, FChunk, Int64(FConfig.Heads) * FConfig.HeadSize]);
  Result := Linear(Context, Prefix + 'o_proj.weight');
end;

function TGemma3nGraphBuilder.Layer(const Hidden, PerLayer: string; const Index: Integer): string;
var
  Prefix, AltUp, Modes, Coefficients, Predicted, Active, Norm, Laurel, Attended, Combined,
    Gate, Center, Delta, Variance, Cutoff, Activated, Innovation, Corrected, PerLayerOut,
    ClippedWeight, First, Last: string;
begin
  Prefix := 'layers.' + IntToStr(Index) + '.';
  AltUp := Prefix + 'altup.';
  Active := Gather(Hidden, IntegerValue(FConfig.AltUpActive), 0);
  Modes := Router(Active, AltUp);
  Coefficients := Linear(Modes, AltUp + 'prediction_coefs.weight');
  Coefficients := Reshape(Coefficients, [1, FChunk, FConfig.AltUpInputs, FConfig.AltUpInputs]);
  Coefficients := Transpose(Coefficients, [0, 1, 3, 2]);
  Predicted := Op('MatMul', [Transpose(Hidden, [1, 2, 3, 0]), Coefficients]);
  Predicted := Op('Add', [Transpose(Predicted, [3, 0, 1, 2]), Hidden]);
  Active := Gather(Predicted, IntegerValue(FConfig.AltUpActive), 0);
  Norm := Normalize(Active, Prefix + 'input_layernorm.weight');
  Laurel := Linear(Linear(Norm, Prefix + 'laurel.linear_left.weight'), Prefix + 'laurel.linear_right.weight');
  Laurel := Op('Add', [Norm, Normalize(Laurel, Prefix + 'laurel.post_laurel_norm.weight')]);
  Attended := Attention(Norm, Prefix + 'self_attn.', Index);
  Attended := Normalize(Attended, Prefix + 'post_attention_layernorm.weight');
  Combined := Op('Mul', [Op('Add', [Op('Add', [Active, Attended]), Laurel]), Float(Sqrt(0.5))]);
  Norm := Normalize(Combined, Prefix + 'pre_feedforward_layernorm.weight');
  Gate := Linear(Norm, Prefix + 'mlp.gate_proj.weight');
  if FConfig.Sparsity[Index] > 0 then
  begin
    Center := Mean(Gate, -1);
    Delta := Op('Sub', [Gate, Center]);
    Variance := Mean(Op('Mul', [Delta, Delta]), -1);
    Cutoff := Op('Add', [Center, Op('Mul', [Op('Sqrt', [Variance]), Float(FConfig.SparsityScale[Index])])]);
    Gate := Op('Relu', [Op('Sub', [Gate, Cutoff])]);
  end;
  Gate := Op('Mul', [GELU(Gate), Linear(Norm, Prefix + 'mlp.up_proj.weight')]);
  Activated := Normalize(Linear(Gate, Prefix + 'mlp.down_proj.weight'),
    Prefix + 'post_feedforward_layernorm.weight');
  Activated := Op('Add', [Combined, Activated]);
  Innovation := Op('Sub', [Activated, Active]);
  Modes := Router(Activated, AltUp);
  ClippedWeight := Op('Clip', [Weight(AltUp + 'correction_coefs.weight'), Float(-120), Float(120)]);
  Coefficients := Op('MatMul', [Modes, Transpose(ClippedWeight, [1, 0])]);
  Coefficients := Op('Add', [Coefficients, Float(1)]);
  Coefficients := Reshape(Transpose(Coefficients, [2, 0, 1]), [FConfig.AltUpInputs, 1, FChunk, 1]);
  Corrected := Op('Add', [Predicted, Op('Mul', [Innovation, Coefficients])]);
  PerLayerOut := Gather(Corrected, IntegerValue(FConfig.AltUpActive), 0);
  if FConfig.AltUpCorrectScale then
    PerLayerOut := Op('Mul', [PerLayerOut, Weight(AltUp + 'correct_output_scale')]);
  PerLayerOut := GELU(Linear(PerLayerOut, Prefix + 'per_layer_input_gate.weight'));
  PerLayerOut := Op('Mul', [PerLayerOut, Gather(PerLayer, IntegerValue(Index), 2)]);
  PerLayerOut := Linear(PerLayerOut, Prefix + 'per_layer_projection.weight');
  PerLayerOut := Normalize(PerLayerOut, Prefix + 'post_per_layer_input_norm.weight');
  First := Part(Corrected, 0, 0, 1);
  Last := Op('Add', [Part(Corrected, 0, 1, FConfig.AltUpInputs), PerLayerOut]);
  Result := AxisOp('Concat', [First, Last], 0);
end;

function TGemma3nGraphBuilder.MatchMagnitude(const Input, Magnitude: string): string;
var
  RMS: string;
begin
  RMS := Mean(Op('Mul', [Input, Input]), -1);
  RMS := Op('Sqrt', [Op('Max', [RMS, Float(1e-5)])]);
  Result := Op('Div', [Op('Mul', [Input, Magnitude]), RMS]);
end;

procedure TGemma3nGraphBuilder.AddCacheOperations;
var
  Positions: TArray<Int64>;
  I: Integer;
  Past, Distances, InRange, Relative, Lowest, Zero: string;
begin
  SetLength(Positions, FContext);
  for I := 0 to High(Positions) do Positions[I] := I;
  Past := Integers(Positions);
  Distances := Op('Sub', [Reshape(FPosition, [FChunk, 1]), Reshape(Past, [1, FContext])]);
  InRange := Op('GreaterOrEqual', [Distances, IntegerValue(0)]);
  Lowest := Float(-3.402823466e38);
  Zero := Float(0);
  FFullMask := Reshape(Op('Where', [InRange, Zero, Lowest]), [1, 1, FChunk, FContext]);
  InRange := Op('And', [InRange, Op('Less', [Distances, IntegerValue(FConfig.SlidingWindow)])]);
  FSlidingMask := Reshape(Op('Where', [InRange, Zero, Lowest]), [1, 1, FChunk, FContext]);
  if FChunk = 1 then
    FCacheMask := Op('Equal', [Past, FPosition])
  else begin
    Relative := Op('Sub', [Past, Gather(FPosition, IntegerValue(0), 0)]);
    FCacheIndices := Op('Clip', [Relative, IntegerValue(0), IntegerValue(FChunk - 1)]);
    FCacheMask := Op('And', [Op('GreaterOrEqual', [Relative, IntegerValue(0)]),
      Op('Less', [Relative, IntegerValue(FChunk)])]);
  end;
  FCacheMask := Reshape(FCacheMask, [1, 1, FContext, 1]);
end;

function TGemma3nGraphBuilder.Build: TOnnxModel;
var
  CacheShape: TArray<Int64>;
  Streams: TArray<string>;
  Embedding, PerLayer, Projection, Magnitude, Hidden, Current, Logits: string;
  I: Integer;
begin
  FPosition := 'position';
  FInputs.Add(ValueInfo('embedding', 1, [1, FChunk, FConfig.Hidden]));
  FInputs.Add(ValueInfo('per_layer_embedding', 1, [1, FChunk, Int64(FConfig.Layers) * FConfig.LayerInput]));
  FInputs.Add(ValueInfo(FPosition, 7, [FChunk]));
  CacheShape := [1, FConfig.KVHeads, FContext, FConfig.HeadSize];
  for I := 0 to 2 * FCacheLayers - 1 do
    FInputs.Add(ValueInfo('cache_' + IntToStr(I), 1, CacheShape));
  AddCacheOperations;
  FFullRotary := Rotary(FPosition, FConfig.RopeFrequency, FChunk, FConfig.HeadSize);
  FSlidingRotary := Rotary(FPosition, FConfig.LocalRopeFrequency, FChunk, FConfig.HeadSize);
  Embedding := Op('Mul', ['embedding', Float(Sqrt(FConfig.Hidden))]);
  PerLayer := Reshape('per_layer_embedding', [1, FChunk, FConfig.Layers, FConfig.LayerInput]);
  PerLayer := Op('Mul', [PerLayer, Float(Sqrt(FConfig.LayerInput))]);
  Projection := Linear(Embedding, 'per_layer_model_projection.weight');
  Projection := Op('Mul', [Projection, Float(1 / Sqrt(FConfig.Hidden))]);
  Projection := Reshape(Projection, [1, FChunk, FConfig.Layers, FConfig.LayerInput]);
  Projection := Normalize(Projection, 'per_layer_projection_norm.weight');
  PerLayer := Op('Mul', [Op('Add', [Projection, PerLayer]), Float(Sqrt(0.5))]);
  SetLength(Streams, FConfig.AltUpInputs);
  Streams[0] := Embedding;
  Magnitude := Op('Sqrt', [Mean(Op('Mul', [Embedding, Embedding]), -1)]);
  for I := 1 to High(Streams) do
    Streams[I] := MatchMagnitude(Linear(Embedding, 'altup_projections.' + IntToStr(I - 1) + '.weight'), Magnitude);
  Hidden := Stack(Streams);
  for I := 0 to FConfig.Layers - 1 do Hidden := Layer(Hidden, PerLayer, I);
  Streams[0] := Gather(Hidden, IntegerValue(0), 0);
  Magnitude := Op('Sqrt', [Mean(Op('Mul', [Streams[0], Streams[0]]), -1)]);
  for I := 1 to High(Streams) do
  begin
    Current := Gather(Hidden, IntegerValue(I), 0);
    Current := Linear(Current, 'altup_unembed_projections.' + IntToStr(I - 1) + '.weight');
    Streams[I] := MatchMagnitude(Current, Magnitude);
  end;
  Current := Normalize(Mean(Stack(Streams), 0, False), 'norm.weight');
  Current := Gather(Current, IntegerValue(FChunk - 1), 1);
  Logits := Linear(Current, 'embed_tokens.weight');
  Logits := Op('Mul', [Op('Tanh', [Op('Div', [Logits, Float(30)])]), Float(30)]);
  AddOutput('logits', Logits, 1, [1, FConfig.Vocab]);
  for I := 0 to FCacheLayers - 1 do
  begin
    AddOutput('updated_' + IntToStr(2 * I), FKeys[I], 1, CacheShape);
    AddOutput('updated_' + IntToStr(2 * I + 1), FValues[I], 1, CacheShape);
  end;
  Result := CreateModel;
end;

function BuildGemma3nTextGraph(const Config: TGemma3nConfig; const Bindings: TArray<TGemma3nBinding>;
  const Context, Chunk: Integer; const Quantized: Boolean): TOnnxModel;
var
  Builder: TGemma3nGraphBuilder;
begin
  Builder := TGemma3nGraphBuilder.Create(Config, Bindings, Context, Chunk, Quantized);
  try
    Result := Builder.Build;
  finally
    Builder.Free;
  end;
end;

end.
