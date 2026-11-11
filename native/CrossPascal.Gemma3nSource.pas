{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nSource                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nSource;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF}, CrossPascal.GGUF;

type
  TGemma3nConfig = record
    Layers: Integer;
    Hidden: Integer;
    LayerInput: Integer;
    Heads: Integer;
    KVHeads: Integer;
    HeadSize: Integer;
    Vocab: Integer;
    LayerVocab: Integer;
    SharedLayers: Integer;
    Context: Integer;
    SlidingWindow: Integer;
    AltUpInputs: Integer;
    AltUpActive: Integer;
    LaurelRank: Integer;
    AltUpCorrectScale: Boolean;
    RMSEpsilon: Single;
    RopeFrequency: Single;
    LocalRopeFrequency: Single;
    Intermediate: TArray<Int64>;
    Sliding: TArray<Boolean>;
    Sparsity: TArray<Double>;
    SparsityScale: TArray<Single>;
  end;

  TGemma3nBinding = record
    Name: string;
    SourceName: string;
    Slice: Integer;
    Shape: TArray<Int64>;
  end;

function ReadGemma3nBindings(const Source: TGGUFCheckpoint;
  out Config: TGemma3nConfig): TArray<TGemma3nBinding>;
procedure WriteGemma3nTokenizer(const Source: TGGUFCheckpoint; const SentencePieceLibrary: string;
  const Dest: TStream);
procedure WriteGemma3nTextWeights(const Source: TGGUFCheckpoint;
  const SourceSHA256, SentencePieceLibrary, Directory: string; const Config: TGemma3nConfig;
  const Bindings: TArray<TGemma3nBinding>);

implementation

uses
  {$IFDEF FPC}Math, SysUtils, Windows{$ELSE}System.Math, System.SysUtils, Winapi.Windows{$ENDIF},
  CrossPascal.Json, CrossPascal.SHA256, TVM.CpuGraph;

function ErrorFunction(Value: Double): Double; cdecl; external 'ucrtbase.dll' name 'erf';

function ReadGemma3nBindings(const Source: TGGUFCheckpoint;
  out Config: TGemma3nConfig): TArray<TGemma3nBinding>;
var
  Bindings: TArray<TGemma3nBinding>;
  SourcePrefix, ModelPrefix: string;
  Tensor: TGGUFTensor;
  Scales: TArray<Single>;
  Layer, Slice, Count: Integer;
  Hidden, LayerInput, QWidth, KVWidth, Intermediate, LaurelRank, AltUpInputs: Int64;

  function Dimension(const Key: string; const Low, High: UInt32): Integer;
  var
    Value: UInt32;
  begin
    Value := Source.Unsigned('gemma3n.' + Key);
    RequireCpu((Value >= Low) and (Value <= High), 'Invalid Gemma 3n dimension: ' + Key);
    Result := Integer(Value);
  end;

  procedure Add(const Name, SourceName: string; const Shape: array of Int64; const Slice: Integer = -1);
  var
    Weight: TGGUFTensor;
    I, Rank: Integer;
  begin
    Weight := Source.Tensor(SourceName);
    Rank := Length(Shape);
    for I := 0 to Rank - 1 do
      RequireCpu(Weight.Dimensions[I] = Shape[Rank - 1 - I], 'Wrong GGUF tensor shape: ' + SourceName);
    if Slice >= 0 then
    begin
      RequireCpu(Weight.Dimensions[Rank] = Config.AltUpInputs - 1, 'Wrong AltUp tensor shape: ' + SourceName);
      Inc(Rank);
    end;
    for I := Rank to High(Weight.Dimensions) do
      RequireCpu(Weight.Dimensions[I] = 1, 'Wrong GGUF tensor rank: ' + SourceName);
    Bindings[Count].Name := Name;
    Bindings[Count].SourceName := SourceName;
    Bindings[Count].Slice := Slice;
    SetLength(Bindings[Count].Shape, Length(Shape));
    for I := 0 to High(Shape) do Bindings[Count].Shape[I] := Shape[I];
    Inc(Count);
  end;

  procedure AddLayer(const Name, SourceName: string; const Shape: array of Int64);
  begin
    Add(ModelPrefix + Name, SourcePrefix + SourceName, Shape);
  end;

begin
  Config := Default(TGemma3nConfig);
  RequireCpu(Source.Text('general.architecture') = 'gemma3n', 'Expected a Gemma 3n GGUF checkpoint');
  Config.Layers := Dimension('block_count', 1, 1024);
  Config.Hidden := Dimension('embedding_length', 1, 65536);
  Config.LayerInput := Dimension('embedding_length_per_layer_input', 1, 65536);
  Config.Heads := Dimension('attention.head_count', 1, 1024);
  Config.KVHeads := Dimension('attention.head_count_kv', 1, Config.Heads);
  Config.HeadSize := Dimension('head_dim', 1, 65536);
  Config.SharedLayers := Dimension('attention.shared_kv_layers', 0, Config.Layers - 1);
  Config.Context := Dimension('context_length', 1, High(Integer));
  Config.SlidingWindow := Dimension('attention.sliding_window', 1, Config.Context);
  Config.AltUpInputs := Dimension('altup.num_inputs', 2, 1024);
  Config.AltUpActive := Dimension('altup.active_idx', 0, Config.AltUpInputs - 1);
  Config.AltUpCorrectScale := Source.Flag('gemma3n.altup.correct_scale');
  Config.RMSEpsilon := Source.Float('gemma3n.attention.layer_norm_rms_epsilon');
  Config.RopeFrequency := Source.Float('gemma3n.rope.freq_base');
  Config.LocalRopeFrequency := Source.Float('gemma3n.rope.freq_base_local');
  Config.Sliding := Source.FlagArray('gemma3n.attention.sliding_window_pattern');
  Scales := Source.FloatArray('gemma3n.activation_sparsity_scale');
  Config.SparsityScale := Scales;
  RequireCpu((Length(Config.Sliding) = Config.Layers) and (Length(Scales) = Config.Layers),
    'Gemma 3n needs attention and sparsity settings for each layer');
  RequireCpu(Config.Heads mod Config.KVHeads = 0, 'Gemma 3n attention heads are not grouped evenly');
  RequireCpu((Config.RMSEpsilon > 0) and (Config.RopeFrequency > 0) and (Config.LocalRopeFrequency > 0),
    'Gemma 3n normalization and RoPE settings must be positive');
  Tensor := Source.Tensor('token_embd.weight');
  RequireCpu(Tensor.Dimensions[1] <= High(Integer), 'Gemma 3n vocabulary is too large');
  Config.Vocab := Tensor.Dimensions[1];
  Tensor := Source.Tensor('per_layer_token_embd.weight');
  RequireCpu(Tensor.Dimensions[1] <= High(Integer), 'Gemma 3n per-layer vocabulary is too large');
  Config.LayerVocab := Tensor.Dimensions[1];
  Tensor := Source.Tensor('blk.0.laurel_l.weight');
  RequireCpu(Tensor.Dimensions[1] <= High(Integer), 'Gemma 3n LAuReL rank is too large');
  Config.LaurelRank := Tensor.Dimensions[1];
  SetLength(Config.Sparsity, Config.Layers);
  SetLength(Config.Intermediate, Config.Layers);
  SetLength(Bindings, Config.Layers * 24 + 5 + 2 * (Config.AltUpInputs - 1));
  Count := 0;
  Hidden := Config.Hidden;
  LayerInput := Config.LayerInput;
  QWidth := Int64(Config.Heads) * Config.HeadSize;
  KVWidth := Int64(Config.KVHeads) * Config.HeadSize;
  LaurelRank := Config.LaurelRank;
  AltUpInputs := Config.AltUpInputs;
  Add('embed_tokens.weight', 'token_embd.weight', [Config.Vocab, Hidden]);
  for Layer := 0 to Config.Layers - 1 do
  begin
    if not IsNan(Scales[Layer]) and not IsInfinite(Scales[Layer]) then
      Config.Sparsity[Layer] := (1 + ErrorFunction(Scales[Layer] / Sqrt(2))) / 2;
    SourcePrefix := 'blk.' + IntToStr(Layer) + '.';
    ModelPrefix := 'layers.' + IntToStr(Layer) + '.';
    Tensor := Source.Tensor(SourcePrefix + 'ffn_gate.weight');
    Intermediate := Tensor.Dimensions[1];
    Config.Intermediate[Layer] := Intermediate;
    AddLayer('self_attn.q_proj.weight', 'attn_q.weight', [QWidth, Hidden]);
    AddLayer('self_attn.k_proj.weight', 'attn_k.weight', [KVWidth, Hidden]);
    AddLayer('self_attn.v_proj.weight', 'attn_v.weight', [KVWidth, Hidden]);
    AddLayer('self_attn.o_proj.weight', 'attn_output.weight', [Hidden, QWidth]);
    AddLayer('self_attn.q_norm.weight', 'attn_q_norm.weight', [Config.HeadSize]);
    AddLayer('self_attn.k_norm.weight', 'attn_k_norm.weight', [Config.HeadSize]);
    AddLayer('mlp.gate_proj.weight', 'ffn_gate.weight', [Intermediate, Hidden]);
    AddLayer('mlp.up_proj.weight', 'ffn_up.weight', [Intermediate, Hidden]);
    AddLayer('mlp.down_proj.weight', 'ffn_down.weight', [Hidden, Intermediate]);
    AddLayer('input_layernorm.weight', 'attn_norm.weight', [Hidden]);
    AddLayer('post_attention_layernorm.weight', 'post_attention_norm.weight', [Hidden]);
    AddLayer('pre_feedforward_layernorm.weight', 'ffn_norm.weight', [Hidden]);
    AddLayer('post_feedforward_layernorm.weight', 'post_ffw_norm.weight', [Hidden]);
    AddLayer('altup.correct_output_scale', 'altup_correct_scale.weight', [Hidden]);
    AddLayer('altup.correction_coefs.weight', 'altup_correct_coef.weight', [AltUpInputs, AltUpInputs]);
    AddLayer('altup.prediction_coefs.weight', 'altup_predict_coef.weight', [Sqr(AltUpInputs), AltUpInputs]);
    AddLayer('altup.modality_router.weight', 'altup_router.weight', [AltUpInputs, Hidden]);
    AddLayer('altup.router_norm.weight', 'altup_router_norm.weight', [Hidden]);
    AddLayer('laurel.linear_left.weight', 'laurel_l.weight', [LaurelRank, Hidden]);
    AddLayer('laurel.linear_right.weight', 'laurel_r.weight', [Hidden, LaurelRank]);
    AddLayer('laurel.post_laurel_norm.weight', 'laurel_post_norm.weight', [Hidden]);
    AddLayer('per_layer_input_gate.weight', 'inp_gate.weight', [LayerInput, Hidden]);
    AddLayer('per_layer_projection.weight', 'proj.weight', [Hidden, LayerInput]);
    AddLayer('post_per_layer_input_norm.weight', 'post_norm.weight', [Hidden]);
  end;
  Add('norm.weight', 'output_norm.weight', [Hidden]);
  Add('embed_tokens_per_layer.weight', 'per_layer_token_embd.weight',
    [Config.LayerVocab, Config.Layers * LayerInput]);
  Add('per_layer_model_projection.weight', 'per_layer_model_proj.weight',
    [Config.Layers * LayerInput, Hidden]);
  Add('per_layer_projection_norm.weight', 'per_layer_proj_norm.weight', [LayerInput]);
  for Slice := 0 to Config.AltUpInputs - 2 do
  begin
    Add('altup_projections.' + IntToStr(Slice) + '.weight', 'altup_proj.weight', [Hidden, Hidden], Slice);
    Add('altup_unembed_projections.' + IntToStr(Slice) + '.weight', 'altup_unembd_proj.weight',
      [Hidden, Hidden], Slice);
  end;
  SetLength(Bindings, Count);
  Result := Bindings;
end;

procedure WriteGemma3nTokenizer(const Source: TGGUFCheckpoint; const SentencePieceLibrary: string;
  const Dest: TStream);
type
  TBuild = function(Pieces: PPAnsiChar; Scores: PSingle; Kinds: PInt32; Count: Integer;
    SpecialIDs: PInt32; AddPrefix: Integer; Output: PAnsiChar; Capacity: Integer): Integer; cdecl;
const
  KeyPrefix = 'tokenizer.ggml.';
  IDNames: array[0..3] of string = ('unknown_token_id', 'bos_token_id', 'eos_token_id', 'padding_token_id');
  DefaultIDs: array[0..3] of UInt32 = (3, 2, 1, 0);
var
  Pieces: TArray<UTF8String>;
  Pointers: TArray<PAnsiChar>;
  Scores: TArray<Single>;
  Kinds: TArray<Int32>;
  IDs: array[0..3] of Int32;
  Empty: AnsiChar;
  Prefix: Integer;
  Buffer: TBytes;
  LibraryHandle: HMODULE;
  Build: TBuild;
  I, Count, Size: Integer;
  RawID: UInt32;
begin
  Pieces := Source.TokenPieces(KeyPrefix + 'tokens');
  Scores := Source.FloatArray(KeyPrefix + 'scores');
  Kinds := Source.IntArray(KeyPrefix + 'token_type');
  Count := Length(Pieces);
  RequireCpu((Count > 0) and (Length(Scores) = Count) and (Length(Kinds) = Count),
    'Gemma 3n tokenizer arrays have different lengths');
  Empty := #0;
  SetLength(Pointers, Count);
  for I := 0 to Count - 1 do
    if Pieces[I] = '' then
      Pointers[I] := @Empty
    else
      Pointers[I] := PAnsiChar(Pieces[I]);
  for I := 0 to High(IDs) do
  begin
    RawID := Source.Unsigned(KeyPrefix + IDNames[I], DefaultIDs[I]);
    if RawID = High(UInt32) then
      IDs[I] := -1
    else begin
      RequireCpu(RawID < UInt32(Count), 'Gemma 3n tokenizer ID is out of range: ' + IDNames[I]);
      IDs[I] := Int32(RawID);
    end;
  end;
  Prefix := Ord(Source.Flag(KeyPrefix + 'add_space_prefix', False));
  LibraryHandle := LoadLibraryExW(PWideChar(ExpandFileName(SentencePieceLibrary)), 0,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR or LOAD_LIBRARY_SEARCH_SYSTEM32);
  if LibraryHandle = 0 then RaiseLastOSError;
  try
    Build := TBuild(GetProcAddress(LibraryHandle, 'cp_sp_build_bpe'));
    RequireCpu(Assigned(Build), 'The SentencePiece library cannot build a BPE tokenizer');
    Size := Build(@Pointers[0], @Scores[0], @Kinds[0], Count, @IDs[0], Prefix, nil, 0);
    RequireCpu(Size > 0, 'Could not build the Gemma 3n tokenizer');
    SetLength(Buffer, Size);
    RequireCpu(Build(@Pointers[0], @Scores[0], @Kinds[0], Count, @IDs[0], Prefix, PAnsiChar(@Buffer[0]), Size) = Size,
      'Could not serialize the Gemma 3n tokenizer');
    Dest.WriteBuffer(Buffer[0], Size);
  finally
    FreeLibrary(LibraryHandle);
  end;
end;

function ShapeJSON(const Shape: array of Int64): TJSONArray;
var
  Dim: Int64;
begin
  Result := TJSONArray.Create;
  for Dim in Shape do Result.AddElement(JSONInteger(Dim));
end;

function Gemma3nConfigJSON(const Config: TGemma3nConfig): TJSONObject;
var
  LayerTypes, Sparsity: TJSONArray;
  I: Integer;
begin
  Result := TJSONObject.Create;
  Result.AddPair('model_type', 'gemma3n_text');
  Result.AddPair('vocab_size', JSONInteger(Config.Vocab));
  Result.AddPair('vocab_size_per_layer_input', JSONInteger(Config.LayerVocab));
  Result.AddPair('hidden_size', JSONInteger(Config.Hidden));
  Result.AddPair('num_hidden_layers', JSONInteger(Config.Layers));
  Result.AddPair('intermediate_size', ShapeJSON(Config.Intermediate));
  Result.AddPair('hidden_size_per_layer_input', JSONInteger(Config.LayerInput));
  Result.AddPair('num_attention_heads', JSONInteger(Config.Heads));
  Result.AddPair('num_key_value_heads', JSONInteger(Config.KVHeads));
  Result.AddPair('head_dim', JSONInteger(Config.HeadSize));
  Result.AddPair('rms_norm_eps', JSONFloat(Config.RMSEpsilon));
  Result.AddPair('sliding_window', JSONInteger(Config.SlidingWindow));
  Result.AddPair('num_kv_shared_layers', JSONInteger(Config.SharedLayers));
  Result.AddPair('altup_num_inputs', JSONInteger(Config.AltUpInputs));
  Result.AddPair('altup_active_idx', JSONInteger(Config.AltUpActive));
  Result.AddPair('altup_correct_scale', JSONBoolean(Config.AltUpCorrectScale));
  Result.AddPair('laurel_rank', JSONInteger(Config.LaurelRank));
  Result.AddPair('max_position_embeddings', JSONInteger(Config.Context));
  Result.AddPair('rope_theta', JSONFloat(Config.RopeFrequency));
  Result.AddPair('rope_local_base_freq', JSONFloat(Config.LocalRopeFrequency));
  LayerTypes := TJSONArray.Create;
  Result.AddPair('layer_types', LayerTypes);
  Sparsity := TJSONArray.Create;
  Result.AddPair('activation_sparsity_pattern', Sparsity);
  for I := 0 to Config.Layers - 1 do
  begin
    if Config.Sliding[I] then
      LayerTypes.AddElement(TJSONString.Create('sliding_attention'))
    else
      LayerTypes.AddElement(TJSONString.Create('full_attention'));
    Sparsity.AddElement(JSONFloat(Config.Sparsity[I]));
  end;
end;

procedure WriteGemma3nTextWeights(const Source: TGGUFCheckpoint;
  const SourceSHA256, SentencePieceLibrary, Directory: string; const Config: TGemma3nConfig;
  const Bindings: TArray<TGemma3nBinding>);
var
  Binding: TGemma3nBinding;
  Tensor: TGGUFTensor;
  RecordData, Weights, Parameters, Entry: TJSONObject;
  Names: TStringList;
  Stream: TFileStream;
  Shape: TArray<Int64>;
  Name, FileName, Path, Hash: string;
  Text: UTF8String;
  I, J, Rank: Integer;
begin
  RequireCpu(DirectoryExists(Directory) and (Length(SourceSHA256) = 64) and (Length(Bindings) > 0),
    'Invalid Gemma 3n parameter destination or source');
  RecordData := TJSONObject.Create;
  Names := nil;
  try
    Names := TStringList.Create;
    Names.CaseSensitive := True;
    Names.Sorted := True;
    Names.Duplicates := dupIgnore;
    RecordData.AddPair('schema', 'crosspascal.gemma3n.parameters.v1');
    RecordData.AddPair('source_sha256', SourceSHA256);
    RecordData.AddPair('source_format', 'GGUF');
    RecordData.AddPair('config', Gemma3nConfigJSON(Config));
    Parameters := TJSONObject.Create;
    RecordData.AddPair('parameters', Parameters);
    Weights := TJSONObject.Create;
    RecordData.AddPair('weights', Weights);
    for I := 0 to High(Bindings) do
    begin
      Binding := Bindings[I];
      Names.AddObject(Binding.SourceName, TObject(NativeInt(I + 1)));
      Entry := TJSONObject.Create;
      Parameters.AddPair(Binding.Name, Entry);
      Entry.AddPair('tensor', Binding.SourceName);
      Entry.AddPair('shape', ShapeJSON(Binding.Shape));
      if Binding.Slice < 0 then
        Entry.AddPair('slice', TJSONNull.Create)
      else
        Entry.AddPair('slice', JSONInteger(Binding.Slice));
    end;
    for I := 0 to Names.Count - 1 do
    begin
      Binding := Bindings[NativeInt(Names.Objects[I]) - 1];
      Name := Binding.SourceName;
      Tensor := Source.Tensor(Name);
      Rank := Length(Binding.Shape) + Ord(Binding.Slice >= 0);
      SetLength(Shape, Rank);
      for J := 0 to High(Shape) do Shape[J] := Tensor.Dimensions[Rank - 1 - J];
      RequireCpu(TensorElements(Shape) = Tensor.Elements, 'Invalid Gemma 3n source tensor shape: ' + Name);
      FileName := Format('weight-%.4d.bin', [I]);
      Path := BundleMember(Directory, FileName);
      Stream := TFileStream.Create(Path, fmCreate);
      try
        Source.WriteFloat32(Name, Stream);
        Stream.Position := 0;
        Hash := SHA256Stream(Stream);
      finally
        Stream.Free;
      end;
      Entry := TJSONObject.Create;
      Weights.AddPair(Name, Entry);
      Entry.AddPair('file', FileName);
      Entry.AddPair('dtype', 'float32');
      Entry.AddPair('shape', ShapeJSON(Shape));
      Entry.AddPair('sha256', Hash);
    end;
    Stream := TFileStream.Create(BundleMember(Directory, 'tokenizer.model'), fmCreate);
    try
      WriteGemma3nTokenizer(Source, SentencePieceLibrary, Stream);
    finally
      Stream.Free;
    end;
    Text := UTF8Encode(RecordData.ToJSON + sLineBreak);
    Stream := TFileStream.Create(BundleMember(Directory, 'parameters.json'), fmCreate);
    try
      Stream.WriteBuffer(Pointer(Text)^, Length(Text));
    finally
      Stream.Free;
    end;
  finally
    Names.Free;
    RecordData.Free;
  end;
end;

end.
