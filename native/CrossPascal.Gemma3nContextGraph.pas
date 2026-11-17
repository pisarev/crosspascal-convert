{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nContextGraph                                            }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nContextGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Onnx;

{ Rebuilds the weightless Gemma 3n text graph of an accepted converted bundle
  for a new context. CPU graphs bind the existing FP32 weights; GPU graphs bind
  the bundle's existing row-wise int8 matrices. The caller owns the result. }
function BuildGemma3nContextGraph(const RecordData, Manifest: TJSONObject; const Context: Integer;
  const Chunk: Integer = 1): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Generics.Collections{$ELSE}System.SysUtils, System.Math, System.Generics.Collections{$ENDIF},
  CrossPascal.Gemma3nSource, CrossPascal.Gemma3nGraph, TVM.CpuGraph;

const
  ParameterSchema = 'crosspascal.gemma3n.parameters.v1';
  ManifestSchema = 'crosspascal.gemma3n.text.v1';
  CpuBackend = 'TVM LLVM CPU FP32';
  GpuBackend = 'TVM CUDA W8A32';
  RowScheme = 'symmetric-int8-per-row';
  MaxContext = 131072;
  WeightPrefix = 'text.';

function ErrorFunction(Value: Double): Double; cdecl; external 'ucrtbase.dll' name 'erf';

function Member(const Obj: TJSONObject; const Name: string): TJSONValue;
begin
  Result := Obj.GetValue(Name);
  RequireCpu(Result <> nil, 'Missing Gemma 3n metadata field: ' + Name);
end;

function TextMember(const Obj: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := Member(Obj, Name);
  RequireCpu((Value is TJSONString) and not (Value is TJSONNumber),
    'Expected text for Gemma 3n field: ' + Name);
  Result := JSONValueText(Value);
end;

function Bounded(const Value: TJSONValue; const Name: string; const Least, Most: Int64): Int64;
begin
  Result := ModelInt(Value, Name);
  RequireCpu((Result >= Least) and (Result <= Most), 'Gemma 3n field is out of range: ' + Name);
end;

function IntMember(const Obj: TJSONObject; const Name: string; const Least, Most: Int64): Integer;
begin
  Result := Integer(Bounded(Member(Obj, Name), Name, Least, Most));
end;

function PositiveSingle(const Obj: TJSONObject; const Name: string): Single;
var
  Value: Double;
begin
  Value := ModelFloat(Member(Obj, Name), Name);
  RequireCpu((Value > 0) and (Value <= MaxSingle), 'Gemma 3n field must be positive: ' + Name);
  Result := Value;
  RequireCpu(Result > 0, 'Gemma 3n field underflows FP32: ' + Name);
end;

function LayerArray(const Obj: TJSONObject; const Name: string; const Count: Integer): TJSONArray;
begin
  Result := ModelArray(Member(Obj, Name), Name);
  RequireCpu(Result.Count = Count, 'Gemma 3n field needs one value per layer: ' + Name);
end;

function IsDigest(const Text: string): Boolean;
var
  I: Integer;
begin
  Result := Length(Text) = 64;
  for I := 1 to Length(Text) do
    if not CharInSet(Text[I], ['0'..'9', 'a'..'f']) then Exit(False);
end;

{ Same expression as the GGUF exporter: (1 + erf(threshold / sqrt 2)) / 2. }
function SparsityProbability(const Threshold: Double): Double;
begin
  Result := (1 + ErrorFunction(Threshold / Sqrt(2))) / 2;
end;

{ Recovers the FP32 normal quantile the exporter started from. The bisection
  result is rounded to FP32, then its two FP32 neighbours are tried as well. }
function SparsityThreshold(const Probability: Double; const Layer: Integer): Single;
var
  Lower, Upper, Middle, Error, BestError: Double;
  Center, Candidate: Single;
  Bits, NeighbourBits: Int32;
  Step, I: Integer;
begin
  RequireCpu((Probability > SparsityProbability(-10)) and (Probability < SparsityProbability(10)),
    'Gemma 3n activation sparsity is out of range at layer ' + IntToStr(Layer));
  Lower := -10;
  Upper := 10;
  for I := 1 to 80 do
  begin
    Middle := (Lower + Upper) / 2;
    if SparsityProbability(Middle) < Probability then
      Lower := Middle
    else
      Upper := Middle;
  end;
  Center := (Lower + Upper) / 2;
  Result := Center;
  BestError := Abs(SparsityProbability(Center) - Probability);
  Move(Center, Bits, SizeOf(Bits));
  for Step := -1 to 1 do
    if Step <> 0 then
    begin
      NeighbourBits := Bits + Step;
      Move(NeighbourBits, Candidate, SizeOf(Candidate));
      if IsNan(Candidate) or IsInfinite(Candidate) then Continue;
      Error := Abs(SparsityProbability(Candidate) - Probability);
      if Error < BestError then
      begin
        Result := Candidate;
        BestError := Error;
      end;
    end;
  RequireCpu(BestError <= 1e-12,
    'Gemma 3n activation sparsity is not an FP32 normal quantile at layer ' + IntToStr(Layer));
end;

{ Reads the fields written by Gemma3nConfigJSON; other keys are ignored. }
procedure ReadConfig(const Data: TJSONObject; out Config: TGemma3nConfig);
var
  Sizes, Types, Pattern: TJSONArray;
  Value: TJSONValue;
  Kind: string;
  Probability: Double;
  I: Integer;
begin
  Config := Default(TGemma3nConfig);
  RequireCpu(TextMember(Data, 'model_type') = 'gemma3n_text', 'Expected a Gemma 3n text configuration');
  Config.Layers := IntMember(Data, 'num_hidden_layers', 1, 1024);
  Config.Hidden := IntMember(Data, 'hidden_size', 1, 65536);
  Config.LayerInput := IntMember(Data, 'hidden_size_per_layer_input', 1, 65536);
  Config.Heads := IntMember(Data, 'num_attention_heads', 1, 1024);
  Config.KVHeads := IntMember(Data, 'num_key_value_heads', 1, Config.Heads);
  Config.HeadSize := IntMember(Data, 'head_dim', 1, 65536);
  Config.Vocab := IntMember(Data, 'vocab_size', 1, High(Integer));
  Config.LayerVocab := IntMember(Data, 'vocab_size_per_layer_input', 1, High(Integer));
  Config.SharedLayers := IntMember(Data, 'num_kv_shared_layers', 0, Config.Layers - 1);
  Config.Context := IntMember(Data, 'max_position_embeddings', 1, High(Integer));
  Config.SlidingWindow := IntMember(Data, 'sliding_window', 1, Config.Context);
  Config.AltUpInputs := IntMember(Data, 'altup_num_inputs', 2, 1024);
  Config.AltUpActive := IntMember(Data, 'altup_active_idx', 0, Config.AltUpInputs - 1);
  Config.LaurelRank := IntMember(Data, 'laurel_rank', 1, High(Integer));
  Value := Member(Data, 'altup_correct_scale');
  RequireCpu(JSONIsBoolean(Value), 'Expected a boolean for Gemma 3n field: altup_correct_scale');
  Config.AltUpCorrectScale := JSONIsTrue(Value);
  Config.RMSEpsilon := PositiveSingle(Data, 'rms_norm_eps');
  Config.RopeFrequency := PositiveSingle(Data, 'rope_theta');
  Config.LocalRopeFrequency := PositiveSingle(Data, 'rope_local_base_freq');
  if Data.GetValue('final_logit_softcapping') <> nil then
    RequireCpu(ModelFloat(Data.GetValue('final_logit_softcapping'), 'logit softcap') = 30,
      'The converted Gemma 3n logit softcap differs from the graph');
  if Data.GetValue('altup_coef_clip') <> nil then
    RequireCpu(ModelFloat(Data.GetValue('altup_coef_clip'), 'AltUp coefficient clip') = 120,
      'The converted Gemma 3n AltUp configuration differs from the graph');
  RequireCpu(Config.Heads mod Config.KVHeads = 0, 'Gemma 3n attention heads are not grouped evenly');
  Sizes := LayerArray(Data, 'intermediate_size', Config.Layers);
  Types := LayerArray(Data, 'layer_types', Config.Layers);
  Pattern := LayerArray(Data, 'activation_sparsity_pattern', Config.Layers);
  SetLength(Config.Intermediate, Config.Layers);
  SetLength(Config.Sliding, Config.Layers);
  SetLength(Config.Sparsity, Config.Layers);
  SetLength(Config.SparsityScale, Config.Layers);
  for I := 0 to Config.Layers - 1 do
  begin
    Config.Intermediate[I] := Bounded(Sizes.Items[I], 'intermediate_size', 1, High(Integer));
    RequireCpu(Types.Items[I] is TJSONString, 'Expected text for Gemma 3n field: layer_types');
    Kind := ModelText(Types.Items[I]);
    RequireCpu((Kind = 'sliding_attention') or (Kind = 'full_attention'),
      'Unknown Gemma 3n layer type: ' + Kind);
    Config.Sliding[I] := Kind = 'sliding_attention';
    Probability := ModelFloat(Pattern.Items[I], 'activation_sparsity_pattern');
    RequireCpu((Probability >= 0) and (Probability < 1),
      'Gemma 3n activation sparsity is out of range at layer ' + IntToStr(I));
    Config.Sparsity[I] := Probability;
    if Probability > 0 then
      Config.SparsityScale[I] := SparsityThreshold(Probability, I)
    else
      Config.SparsityScale[I] := 0;
  end;
end;

function ReadBindings(const Parameters: TJSONObject; const Config: TGemma3nConfig): TArray<TGemma3nBinding>;
var
  Bindings: TArray<TGemma3nBinding>;
  Index: TDictionary<string, Integer>;
  Entry: TJSONObject;
  Slice: TJSONValue;
  Name, Prefix: string;
  Hidden, LayerInput, QWidth, KVWidth, AltUp, Rank, Inner: Int64;
  Count, I, Layer: Integer;

  procedure Expect(const Key: string; const Shape: array of Int64; const SliceIndex: Integer = -1);
  var
    Position, J: Integer;
  begin
    RequireCpu(Index.TryGetValue(Key, Position), 'Missing Gemma 3n parameter: ' + Key);
    RequireCpu((Length(Bindings[Position].Shape) = Length(Shape)) and
      (Bindings[Position].Slice = SliceIndex), 'Wrong Gemma 3n parameter layout: ' + Key);
    for J := 0 to High(Shape) do
      RequireCpu(Bindings[Position].Shape[J] = Shape[J], 'Wrong Gemma 3n parameter shape: ' + Key);
    Inc(Count);
  end;

begin
  Index := TDictionary<string, Integer>.Create;
  try
    SetLength(Bindings, Parameters.Count);
    for I := 0 to Parameters.Count - 1 do
    begin
      Name := JSONObjectName(Parameters, I);
      RequireCpu((Name <> '') and (Copy(Name, 1, Length(WeightPrefix)) <> WeightPrefix),
        'Invalid Gemma 3n parameter name: ' + Name);
      RequireCpu(not Index.ContainsKey(Name), 'Duplicate Gemma 3n parameter: ' + Name);
      Index.Add(Name, I);
      Entry := ModelObject(JSONObjectValue(Parameters, I), Name);
      Bindings[I].Name := Name;
      Bindings[I].SourceName := TextMember(Entry, 'tensor');
      RequireCpu(Bindings[I].SourceName <> '', 'Missing Gemma 3n source tensor: ' + Name);
      Bindings[I].SourceFormat := 0;
      Bindings[I].Shape := ModelShape(Member(Entry, 'shape'));
      Slice := Member(Entry, 'slice');
      if Slice is TJSONNull then
        Bindings[I].Slice := -1
      else
        Bindings[I].Slice := Integer(Bounded(Slice, Name + '.slice', 0, Config.AltUpInputs - 2));
    end;
    Count := 0;
    Hidden := Config.Hidden;
    LayerInput := Config.LayerInput;
    QWidth := Int64(Config.Heads) * Config.HeadSize;
    KVWidth := Int64(Config.KVHeads) * Config.HeadSize;
    AltUp := Config.AltUpInputs;
    Rank := Config.LaurelRank;
    Expect('embed_tokens.weight', [Config.Vocab, Hidden]);
    for Layer := 0 to Config.Layers - 1 do
    begin
      Prefix := 'layers.' + IntToStr(Layer) + '.';
      Inner := Config.Intermediate[Layer];
      Expect(Prefix + 'self_attn.q_proj.weight', [QWidth, Hidden]);
      Expect(Prefix + 'self_attn.k_proj.weight', [KVWidth, Hidden]);
      Expect(Prefix + 'self_attn.v_proj.weight', [KVWidth, Hidden]);
      Expect(Prefix + 'self_attn.o_proj.weight', [Hidden, QWidth]);
      Expect(Prefix + 'self_attn.q_norm.weight', [Config.HeadSize]);
      Expect(Prefix + 'self_attn.k_norm.weight', [Config.HeadSize]);
      Expect(Prefix + 'mlp.gate_proj.weight', [Inner, Hidden]);
      Expect(Prefix + 'mlp.up_proj.weight', [Inner, Hidden]);
      Expect(Prefix + 'mlp.down_proj.weight', [Hidden, Inner]);
      Expect(Prefix + 'input_layernorm.weight', [Hidden]);
      Expect(Prefix + 'post_attention_layernorm.weight', [Hidden]);
      Expect(Prefix + 'pre_feedforward_layernorm.weight', [Hidden]);
      Expect(Prefix + 'post_feedforward_layernorm.weight', [Hidden]);
      Expect(Prefix + 'altup.correct_output_scale', [Hidden]);
      Expect(Prefix + 'altup.correction_coefs.weight', [AltUp, AltUp]);
      Expect(Prefix + 'altup.prediction_coefs.weight', [AltUp * AltUp, AltUp]);
      Expect(Prefix + 'altup.modality_router.weight', [AltUp, Hidden]);
      Expect(Prefix + 'altup.router_norm.weight', [Hidden]);
      Expect(Prefix + 'laurel.linear_left.weight', [Rank, Hidden]);
      Expect(Prefix + 'laurel.linear_right.weight', [Hidden, Rank]);
      Expect(Prefix + 'laurel.post_laurel_norm.weight', [Hidden]);
      Expect(Prefix + 'per_layer_input_gate.weight', [LayerInput, Hidden]);
      Expect(Prefix + 'per_layer_projection.weight', [Hidden, LayerInput]);
      Expect(Prefix + 'post_per_layer_input_norm.weight', [Hidden]);
    end;
    Expect('norm.weight', [Hidden]);
    Expect('embed_tokens_per_layer.weight', [Config.LayerVocab, Config.Layers * LayerInput]);
    Expect('per_layer_model_projection.weight', [Config.Layers * LayerInput, Hidden]);
    Expect('per_layer_projection_norm.weight', [LayerInput]);
    for I := 0 to Config.AltUpInputs - 2 do
    begin
      Expect('altup_projections.' + IntToStr(I) + '.weight', [Hidden, Hidden], I);
      Expect('altup_unembed_projections.' + IntToStr(I) + '.weight', [Hidden, Hidden], I);
    end;
    RequireCpu(Count = Length(Bindings), 'Unexpected Gemma 3n parameters');
  finally
    Index.Free;
  end;
  Result := Bindings;
end;

function ShapeValue(const Name: string; const Kind: Integer; const Shape: array of Int64): TOnnxValue;
var
  I: Integer;
begin
  Result := Default(TOnnxValue);
  Result.Name := Name;
  Result.DataType := Kind;
  Result.HasShape := True;
  SetLength(Result.Shape, Length(Shape));
  for I := 0 to High(Shape) do Result.Shape[I].Value := Shape[I];
end;

function SameShape(const Value: TOnnxValue; const Shape: array of Int64): Boolean;
var
  I: Integer;
begin
  Result := Value.HasShape and (Length(Value.Shape) = Length(Shape));
  if Result then
    for I := 0 to High(Shape) do
      if Value.Shape[I].Value <> Shape[I] then Exit(False);
end;

function IsRowTranspose(const Node: TOnnxNode): Boolean;
begin
  Result := (Node.OpType = 'Transpose') and (Node.Domain = '') and (Length(Node.Inputs) = 1) and
    (Length(Node.Outputs) = 1) and (Length(Node.Attributes) = 1) and
    (Node.Attributes[0].Name = 'perm') and (Length(Node.Attributes[0].IntValues) = 2) and
    (Node.Attributes[0].IntValues[0] = 1) and (Node.Attributes[0].IntValues[1] = 0);
end;

{ Rewrites MatMul(X, Transpose(W, [1, 0])) into crosspascal:Q8MatMul(X, W_q8, W_scale)
  for every W listed in the bundle's row-wise int8 table. }
function BindRowQ8(const Graph: TOnnxModel; const Manifest: TJSONObject): TOnnxModel;
var
  Weights: TDictionary<string, TArray<Int64>>;
  Producers, InputIndex, Readers, Rewritten: TDictionary<string, Integer>;
  Inputs, Appended: TList<TOnnxValue>;
  Nodes, Kept: TArray<TOnnxNode>;
  Touched, Removed: TArray<Boolean>;
  Initializers: TArray<TOnnxTensor>;
  Owned: TArray<TBytes>;
  Quantization, Table, Entry: TJSONObject;
  Value: TOnnxValue;
  Shape: TArray<Int64>;
  Name, Source: string;
  I, J, Producer, Count: Integer;

  function ReaderCount(const Key: string): Integer;
  begin
    if not Readers.TryGetValue(Key, Result) then Result := 0;
  end;

  procedure Read(const Key: string; const Delta: Integer);
  begin
    Readers.AddOrSetValue(Key, ReaderCount(Key) + Delta);
  end;

  procedure Define(const Key: string; const Index: Integer);
  begin
    RequireCpu(not Producers.ContainsKey(Key) and not InputIndex.ContainsKey(Key),
      'Duplicate ONNX value: ' + Key);
    Producers.Add(Key, Index);
  end;

  procedure AddPair(const List: TList<TOnnxValue>; const Key: string; const Rows: TArray<Int64>);
  begin
    List.Add(ShapeValue(Key + '_q8', OnnxInt8, Rows));
    List.Add(ShapeValue(Key + '_scale', OnnxFloat, [Rows[0]]));
  end;

begin
  Weights := nil;
  Producers := nil;
  InputIndex := nil;
  Readers := nil;
  Rewritten := nil;
  Inputs := nil;
  Appended := nil;
  try
    Weights := TDictionary<string, TArray<Int64>>.Create;
    Producers := TDictionary<string, Integer>.Create;
    InputIndex := TDictionary<string, Integer>.Create;
    Readers := TDictionary<string, Integer>.Create;
    Rewritten := TDictionary<string, Integer>.Create;
    Inputs := TList<TOnnxValue>.Create;
    Appended := TList<TOnnxValue>.Create;
    Quantization := ModelObject(Member(Manifest, 'quantization'), 'quantization');
    RequireCpu(TextMember(Quantization, 'scheme') = RowScheme, 'Expected row-wise int8 Gemma 3n weights');
    RequireCpu(TextMember(Quantization, 'activation_dtype') = 'float32',
      'Expected FP32 Gemma 3n activations');
    Table := ModelObject(Member(Quantization, 'weights'), 'quantization.weights');
    RequireCpu(Table.Count > 0, 'The Gemma 3n GPU bundle has no quantized weights');
    for I := 0 to Table.Count - 1 do
    begin
      Name := JSONObjectName(Table, I);
      RequireCpu(Copy(Name, 1, Length(WeightPrefix)) = WeightPrefix,
        'Invalid quantized weight name: ' + Name);
      RequireCpu(not Weights.ContainsKey(Name), 'Duplicate quantized weight: ' + Name);
      Entry := ModelObject(JSONObjectValue(Table, I), Name);
      RequireCpu((TextMember(Entry, 'q8') <> '') and (TextMember(Entry, 'scale') <> ''),
        'Incomplete quantized weight: ' + Name);
      Shape := ModelShape(Member(Entry, 'shape'));
      RequireCpu(Length(Shape) = 2, 'Quantized weight must be a matrix: ' + Name);
      Weights.Add(Name, Shape);
    end;
    for I := 0 to High(Graph.Inputs) do
    begin
      Name := Graph.Inputs[I].Name;
      RequireCpu(not InputIndex.ContainsKey(Name), 'Duplicate ONNX input: ' + Name);
      InputIndex.Add(Name, I);
    end;
    for I := 0 to High(Graph.Initializers) do Define(Graph.Initializers[I].Name, -1);
    Nodes := Copy(Graph.Nodes);
    for I := 0 to High(Nodes) do
    begin
      for Name in Nodes[I].Inputs do Read(Name, 1);
      for Name in Nodes[I].Outputs do Define(Name, I);
    end;
    for Value in Graph.Outputs do Read(Value.Name, 1);
    SetLength(Touched, Length(Nodes));
    SetLength(Removed, Length(Nodes));
    for I := 0 to High(Nodes) do
    begin
      if (Nodes[I].OpType <> 'MatMul') or (Nodes[I].Domain <> '') or (Length(Nodes[I].Inputs) <> 2) or
        (Length(Nodes[I].Attributes) <> 0) then
          Continue;
      if not Producers.TryGetValue(Nodes[I].Inputs[1], Producer) or (Producer < 0) or
        not IsRowTranspose(Nodes[Producer]) then Continue;
      Name := Nodes[Producer].Inputs[0];
      if not Weights.TryGetValue(Name, Shape) then Continue;
      RequireCpu(InputIndex.TryGetValue(Name, J) and (Graph.Inputs[J].DataType = OnnxFloat) and SameShape(Graph.Inputs[J], Shape),
        'Quantized Gemma 3n weight does not match the graph: ' + Name);
      if not Rewritten.ContainsKey(Name) then
      begin
        Define(Name + '_q8', -1);
        Define(Name + '_scale', -1);
        Rewritten.Add(Name, 0);
      end;
      Rewritten[Name] := Rewritten[Name] + 1;
      Source := Nodes[I].Inputs[0];
      Read(Nodes[I].Inputs[1], -1);
      Nodes[I].OpType := 'Q8MatMul';
      Nodes[I].Domain := 'crosspascal';
      Nodes[I].Inputs := [Source, Name + '_q8', Name + '_scale'];
      Read(Name + '_q8', 1);
      Read(Name + '_scale', 1);
      Touched[Producer] := True;
    end;
    RequireCpu(Rewritten.Count = Weights.Count,
      'Some quantized Gemma 3n weights are not graph projections');
    for I := 0 to High(Nodes) do
      if Touched[I] and (ReaderCount(Nodes[I].Outputs[0]) = 0) then
      begin
        Removed[I] := True;
        Read(Nodes[I].Inputs[0], -1);
      end;
    for I := 0 to High(Graph.Inputs) do
    begin
      Value := Graph.Inputs[I];
      if Rewritten.ContainsKey(Value.Name) then
      begin
        Shape := Weights[Value.Name];
        if ReaderCount(Value.Name) = 0 then
          AddPair(Inputs, Value.Name, Shape)
        else begin
          Inputs.Add(Value);
          AddPair(Appended, Value.Name, Shape);
        end;
      end
      else begin
        RequireCpu(ReaderCount(Value.Name) > 0, 'Unused ONNX input: ' + Value.Name);
        Inputs.Add(Value);
      end;
    end;
    for Value in Appended do Inputs.Add(Value);
    SetLength(Kept, Length(Nodes));
    Count := 0;
    for I := 0 to High(Nodes) do
      if not Removed[I] then
      begin
        Kept[Count] := Nodes[I];
        Inc(Count);
      end;
    SetLength(Kept, Count);
    { Generated constants point into the source model's private buffers, so the
      same bytes are copied into buffers owned by the new model. }
    Initializers := Copy(Graph.Initializers);
    SetLength(Owned, Length(Initializers));
    for I := 0 to High(Initializers) do
    begin
      RequireCpu(not Initializers[I].IsExternal and (Initializers[I].Serialized.Size = 0) and
        (Initializers[I].RawData.Data <> nil) and
        (Copy(Initializers[I].Name, 1, Length(WeightPrefix)) <> WeightPrefix),
        'Unexpected ONNX initializer: ' + Initializers[I].Name);
      SetLength(Owned[I], NativeInt(Initializers[I].RawData.Size));
      Move(Initializers[I].RawData.Data^, Owned[I][0], Initializers[I].RawData.Size);
      Initializers[I].RawData.Data := Pointer(Owned[I]);
    end;
    Result := TOnnxModel.CreateGenerated(Integer(Graph.Opset), Inputs.ToArray, Copy(Graph.Outputs),
      Initializers, Kept, Owned);
  finally
    Appended.Free;
    Inputs.Free;
    Rewritten.Free;
    Readers.Free;
    InputIndex.Free;
    Producers.Free;
    Weights.Free;
  end;
end;

function BuildGemma3nContextGraph(const RecordData, Manifest: TJSONObject;
  const Context, Chunk: Integer): TOnnxModel;
var
  Config: TGemma3nConfig;
  Bindings: TArray<TGemma3nBinding>;
  Backend, Source: string;
  Graph: TOnnxModel;
begin
  RequireCpu((RecordData <> nil) and (Manifest <> nil), 'Missing Gemma 3n metadata');
  RequireCpu(TextMember(RecordData, 'schema') = ParameterSchema, 'Unsupported Gemma 3n parameter schema');
  RequireCpu(TextMember(Manifest, 'schema') = ManifestSchema, 'Unsupported Gemma 3n model schema');
  RequireCpu(TextMember(RecordData, 'source_format') = 'GGUF', 'Unsupported Gemma 3n source format');
  Source := TextMember(RecordData, 'source_sha256');
  RequireCpu(IsDigest(Source) and (TextMember(Manifest, 'source_sha256') = Source),
    'Gemma 3n parameters and model come from different sources');
  Backend := TextMember(Manifest, 'backend');
  RequireCpu((Backend = CpuBackend) or (Backend = GpuBackend), 'Unsupported Gemma 3n backend: ' + Backend);
  ReadConfig(ModelObject(Member(RecordData, 'config'), 'config'), Config);
  RequireCpu((Context >= 2) and (Context <= MaxContext) and (Context <= Config.Context),
    'Gemma 3n context is out of range');
  Bindings := ReadBindings(ModelObject(Member(RecordData, 'parameters'), 'parameters'), Config);
  RequireCpu((Chunk >= 1) and (Chunk <= Context), 'Invalid Gemma 3n prefill chunk');
  Graph := BuildGemma3nTextGraph(Config, Bindings, Context, Chunk, False);
  if Backend = CpuBackend then Exit(Graph);
  try
    Result := BindRowQ8(Graph, Manifest);
  finally
    Graph.Free;
  end;
end;

end.
