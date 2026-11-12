{ ************************************************************************** }
{                                                                            }
{ ConvertQwenVLProfile                                                       }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertQwenVLProfile;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils, Classes, Windows, Generics.Collections
  {$ELSE}System.SysUtils, System.Classes, Winapi.Windows, System.Generics.Collections{$ENDIF},
  CrossPascal.Json, CrossPascal.Onnx, CrossPascal.RowQuantization, CrossPascal.SafeTensors,
  CrossPascal.SHA256, CrossPascal.TVMCompiler, CrossPascal.TVMOnnx, TVM.CpuGraph;

procedure Check(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise Exception.Create(MessageText);
end;

function FileIn(const Directory, Name: string): string;
begin
  Check((Name <> '') and (Name <> '.') and (Name <> '..') and
    (Pos('/', Name) = 0) and (Pos('\', Name) = 0) and (Pos(':', Name) = 0),
    'Invalid profile filename: ' + Name);
  Result := IncludeTrailingPathDelimiter(Directory) + Name;
end;

function ObjectValue(const Value: TJSONValue; const LabelText: string): TJSONObject;
begin
  Check(Value is TJSONObject, 'Expected an object for ' + LabelText);
  Result := TJSONObject(Value);
end;

function StringValue(const Obj: TJSONObject; const Key: string): string;
var
  Value: TJSONValue;
begin
  Value := Obj.GetValue(Key);
  Check(Value is TJSONString, 'Missing profile string: ' + Key);
  Result := JSONValueText(Value);
end;

function ReadObject(const FileName: string): TJSONObject;
var
  Stream: TFileStream;
  Raw: UTF8String;
  Value: TJSONValue;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    Check((Stream.Size > 0) and (Stream.Size <= 64 * 1024 * 1024), 'Invalid profile metadata size');
    SetLength(Raw, Stream.Size);
    Stream.ReadBuffer(Raw[1], Length(Raw));
  finally
    Stream.Free;
  end;
  Value := ParseCrossJSON(UTF8Decode(Raw));
  if not (Value is TJSONObject) then
  begin
    Value.Free;
    raise Exception.Create('Invalid JSON object: ' + FileName);
  end;
  Result := TJSONObject(Value);
end;

function FileHash(const FileName: string): string;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    Result := SHA256Stream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure CheckHash(const FileName, Expected: string);
begin
  Check(SameText(FileHash(FileName), Expected), 'Hash mismatch: ' + FileName);
end;

procedure CopyChecked(const Source, Target, Expected: string);
var
  Input, Output: TFileStream;
begin
  CheckHash(Source, Expected);
  Input := TFileStream.Create(Source, fmOpenRead or fmShareDenyWrite);
  try
    Output := TFileStream.Create(Target, fmCreate or fmShareExclusive);
    try
      Output.CopyFrom(Input, 0);
    finally
      Output.Free;
    end;
  finally
    Input.Free;
  end;
  CheckHash(Target, Expected);
end;

function OriginalName(const Name: string; const CheckpointNames: TStringList): string;
begin
  if CheckpointNames.IndexOf(Name) >= 0 then Exit(Name);
  if Name = 'lm_head.weight' then Exit('model.embed_tokens.weight');
  if Copy(Name, 1, 21) = 'model.language_model.' then
    Exit('model.' + Copy(Name, 22, Length(Name)));
  if Copy(Name, 1, 13) = 'model.visual.' then
    Exit(Copy(Name, 7, Length(Name)));
  raise Exception.Create('Unknown Qwen-VL weight name: ' + Name);
end;

procedure ExportWeights(const Checkpoint: TSafeTensorCheckpoint; const Specs: TJSONObject;
  const Destination: string; const Written, SourceNames: TStringList);
var
  Name, Original, Target, Expected: string;
  Spec: TJSONObject;
  Info: TSafeTensorInfo;
  Shape: TJSONArray;
  Value: TJSONValue;
  Output: TFileStream;
  I, J: Integer;
begin
  for I := 0 to Specs.Count - 1 do
  begin
    Name := JSONObjectName(Specs, I);
    Spec := ObjectValue(JSONObjectValue(Specs, I), Name);
    Original := OriginalName(Name, SourceNames);
    Info := Checkpoint.TensorInfo(Original);
    Value := Spec.GetValue('shape');
    Check(Value is TJSONArray, 'Missing target weight shape: ' + Name);
    Shape := TJSONArray(Value);
    Check((Shape.Count = Length(Info.Shape)) and (StringValue(Spec, 'dtype') = 'float32'),
      'Invalid profile weight shape or dtype: ' + Name);
    for J := 0 to Shape.Count - 1 do
      Check(JSONValueText(Shape.Items[J]) = IntToStr(Info.Shape[J]), 'Source weight shape differs: ' + Name);
    Target := StringValue(Spec, 'file');
    Expected := StringValue(Spec, 'sha256');
    J := Written.IndexOfName(Target);
    if J >= 0 then
    begin
      Check(Written.ValueFromIndex[J] = Expected, 'Conflicting profile weight aliases: ' + Target);
      Continue;
    end;
    Output := TFileStream.Create(FileIn(Destination, Target), fmCreate or fmShareExclusive);
    try
      Checkpoint.WriteFloat32(Original, Output);
    finally
      Output.Free;
    end;
    CheckHash(FileIn(Destination, Target), Expected);
    Written.Values[Target] := Expected;
    if Written.Count mod 100 = 0 then WriteLn('Converted ', Written.Count, ' weight files');
  end;
end;

procedure CopyGraphBuffers(const Model: TJSONObject; const Profile, Destination: string;
  const Written: TStringList);
var
  Graphs, Graph, Parameters, Spec: TJSONObject;
  Name, Expected: string;
  I, J, At: Integer;
begin
  Graphs := ObjectValue(Model.GetValue('graphs'), 'graphs');
  for I := 0 to Graphs.Count - 1 do
  begin
    Graph := ObjectValue(JSONObjectValue(Graphs, I), 'graph');
    Parameters := ObjectValue(Graph.GetValue('parameters'), 'graph parameters');
    for J := 0 to Parameters.Count - 1 do
    begin
      Spec := ObjectValue(JSONObjectValue(Parameters, J), 'graph parameter');
      Name := StringValue(Spec, 'file');
      Expected := StringValue(Spec, 'sha256');
      At := Written.IndexOfName(Name);
      if At >= 0 then
      begin
        Check(Written.ValueFromIndex[At] = Expected, 'Graph references different bytes: ' + Name);
        Continue;
      end;
      Check(Pos('-buffer-', Name) > 0,
        'Graph requests a weight absent from the original checkpoint: ' + Name);
      CopyChecked(FileIn(Profile, Name), FileIn(Destination, Name), Expected);
      Written.Values[Name] := Expected;
    end;
  end;
end;

procedure DeleteField(const Obj: TJSONObject; const Name: string);
begin
  if Obj.GetValue(Name) = nil then Exit;
  {$IFDEF FPC}
  Obj.Delete(Obj.IndexOfName(UTF8Encode(Name)));
  {$ELSE}
  Obj.RemovePair(Name).Free;
  {$ENDIF}
end;

procedure AddParameter(const Params: TJSONObject; const Name, FileName, Kind, Hash: string;
  const Dimensions: array of Int64);
var
  Spec: TJSONObject;
  Shape: TJSONArray;
  Size: Int64;
begin
  Spec := TJSONObject.Create;
  Params.AddPair(Name, Spec);
  Shape := TJSONArray.Create;
  Spec.AddPair('shape', Shape);
  for Size in Dimensions do Shape.AddElement(JSONInteger(Size));
  Spec.AddPair('file', FileName);
  Spec.AddPair('dtype', Kind);
  Spec.AddPair('sha256', Hash);
end;

procedure QuantizeMatrix(const Stage, Original, Quantized, Scales: string;
  const Rows, Columns: Integer; const Written: TStringList);
var
  Source: TMappedModelFile;
  Q8File, ScaleFile: TFileStream;
  Size: Int64;
begin
  if Written.IndexOfName(Quantized) >= 0 then Exit;
  Source := TMappedModelFile.Create(FileIn(Stage, Original));
  Q8File := nil;
  ScaleFile := nil;
  try
    Size := Int64(Rows) * Columns * SizeOf(Single);
    Check(Source.Size = Size, 'Invalid matrix size: ' + Original);
    Q8File := TFileStream.Create(FileIn(Stage, Quantized), fmCreate or fmShareExclusive);
    ScaleFile := TFileStream.Create(FileIn(Stage, Scales), fmCreate or fmShareExclusive);
    WriteSymmetricInt8Rows(Source.AtOffset(0, Size), Rows, Columns, Q8File, ScaleFile);
    Q8File.Position := 0;
    ScaleFile.Position := 0;
    Written.Values[Quantized] := SHA256Stream(Q8File);
    Written.Values[Scales] := SHA256Stream(ScaleFile);
  finally
    ScaleFile.Free;
    Q8File.Free;
    Source.Free;
  end;
end;

function QuantizeTextGraph(const Model: TOnnxModel; const Stage: string; const Spec: TJSONObject;
  const Written: TStringList): TOnnxModel;
var
  Matrices, Replaced: TDictionary<string, TOnnxValue>;
  Transposes: TDictionary<string, string>;
  Nodes: TList<TOnnxNode>;
  Inputs: TList<TOnnxValue>;
  Input, Quantized, Scale: TOnnxValue;
  Node, OriginalNode: TOnnxNode;
  Attr: TOnnxAttribute;
  Params, NewParams, Weight: TJSONObject;
  Names: TJSONArray;
  Name, Original, Q8File, ScaleFile: string;
  Rows, Columns, I: Integer;
begin
  Matrices := TDictionary<string, TOnnxValue>.Create;
  Replaced := TDictionary<string, TOnnxValue>.Create;
  Transposes := TDictionary<string, string>.Create;
  Nodes := TList<TOnnxNode>.Create;
  Inputs := TList<TOnnxValue>.Create;
  NewParams := TJSONObject.Create;
  Names := TJSONArray.Create;
  try
    for Input in Model.Inputs do
      if (Input.DataType = OnnxFloat) and (Length(Input.Shape) = 2) and (Copy(Input.Name, Length(Input.Name) - 6, 7) = '.weight') then
        Matrices.Add(Input.Name, Input);
    for Node in Model.Nodes do
      if (Node.OpType = 'Transpose') and (Length(Node.Inputs) = 1) and
        (Length(Node.Outputs) = 1) and Matrices.TryGetValue(Node.Inputs[0], Input) then
          for Attr in Node.Attributes do
            if (Attr.Name = 'perm') and (Length(Attr.IntValues) = 2) and
              (Attr.IntValues[0] = 1) and (Attr.IntValues[1] = 0) then
              begin
                Transposes.Add(Node.Outputs[0], Input.Name);
                Replaced.AddOrSetValue(Input.Name, Input);
              end;
    Check(Replaced.Count > 0, 'Qwen-VL text graph has no quantizable linear weights');
    for OriginalNode in Model.Nodes do
    begin
      Node := OriginalNode;
      if (Node.OpType = 'Transpose') and (Length(Node.Outputs) = 1) and
        Transposes.ContainsKey(Node.Outputs[0]) then Continue;
      for I := 0 to High(Node.Inputs) do
      begin
        Check(not Replaced.ContainsKey(Node.Inputs[I]), 'Unexpected direct use of a quantized weight');
        if not Transposes.TryGetValue(Node.Inputs[I], Name) then Continue;
        Check((Node.OpType = 'MatMul') and (I = 1) and (Length(Node.Inputs) = 2),
          'A transposed Qwen-VL matrix is not a linear projection');
        Node.OpType := 'Q8MatMul';
        Node.Domain := 'crosspascal';
        Node.Inputs := TArray<string>.Create(Node.Inputs[0], Name + ':int8', Name + ':scale');
        Break;
      end;
      Nodes.Add(Node);
    end;
    Params := ObjectValue(Spec.GetValue('parameters'), 'graph parameters');
    for Input in Model.Inputs do
    begin
      if not Replaced.ContainsKey(Input.Name) then
      begin
        Inputs.Add(Input);
        Names.AddElement(TJSONString.Create(Input.Name));
        if Params.GetValue(Input.Name) <> nil then
          NewParams.AddPair(Input.Name,
            ParseCrossJSON(ObjectValue(Params.GetValue(Input.Name), Input.Name).ToJSON));
        Continue;
      end;
      Weight := ObjectValue(Params.GetValue(Input.Name), Input.Name);
      Check((Input.Shape[0].Value > 0) and (Input.Shape[0].Value <= High(Integer)) and
        (Input.Shape[1].Value > 0) and (Input.Shape[1].Value <= High(Integer)),
        'Invalid Qwen-VL linear weight dimensions');
      Rows := Input.Shape[0].Value;
      Columns := Input.Shape[1].Value;
      Original := StringValue(Weight, 'file');
      Q8File := 'q8-' + Original;
      ScaleFile := 'scale-' + Original;
      QuantizeMatrix(Stage, Original, Q8File, ScaleFile, Rows, Columns, Written);
      Quantized := Input;
      Quantized.Name := Input.Name + ':int8';
      Quantized.DataType := OnnxInt8;
      Scale := Input;
      Scale.Name := Input.Name + ':scale';
      Scale.Shape := Copy(Input.Shape, 0, 1);
      Inputs.Add(Quantized);
      Inputs.Add(Scale);
      Names.AddElement(TJSONString.Create(Quantized.Name));
      Names.AddElement(TJSONString.Create(Scale.Name));
      AddParameter(NewParams, Quantized.Name, Q8File, 'int8', Written.Values[Q8File], [Rows, Columns]);
      AddParameter(NewParams, Scale.Name, ScaleFile, 'float32', Written.Values[ScaleFile], [Rows]);
    end;
    Result := TOnnxModel.CreateGenerated(Model.Opset, Inputs.ToArray, Model.Outputs,
      Model.Initializers, Nodes.ToArray, nil);
    DeleteField(Spec, 'input_names');
    Spec.AddPair('input_names', Names);
    Names := nil;
    DeleteField(Spec, 'parameters');
    Spec.AddPair('parameters', NewParams);
    NewParams := nil;
  finally
    Names.Free;
    NewParams.Free;
    Inputs.Free;
    Nodes.Free;
    Transposes.Free;
    Replaced.Free;
    Matrices.Free;
  end;
end;

procedure CompileGraph(const Stage: string; const Spec, Artifacts: TJSONObject;
  const ExpectedLibrary: string; const Kit: TWindowsTvmKit;
  const NvrtcLibrary, Architecture: string; const Quantize: Boolean; const Written: TStringList);
var
  Model, Lowered: TOnnxModel;
  Name, LibraryFile: string;
  Inputs: TJSONValue;
  I: Integer;
begin
  Name := StringValue(Spec, 'library');
  Check(Name = ExpectedLibrary, 'Unexpected Qwen-VL library: ' + Name);
  DeleteField(Artifacts, Name);
  if Architecture <> '' then
  begin
    Name := StringReplace(Name, '-cpu.dll', '-cuda.dll', []);
    DeleteField(Spec, 'library');
    Spec.AddPair('library', Name);
  end;
  LibraryFile := FileIn(Stage, Name);
  Model := TOnnxModel.Create(FileIn(Stage, StringValue(Spec, 'onnx')));
  Lowered := nil;
  try
    Inputs := Spec.GetValue('input_names');
    Check((Inputs is TJSONArray) and (TJSONArray(Inputs).Count = Length(Model.Inputs)),
      'Qwen-VL graph inputs differ from its profile');
    for I := 0 to High(Model.Inputs) do
      Check(Model.Inputs[I].Name = JSONValueText(TJSONArray(Inputs).Items[I]),
        'Unexpected Qwen-VL graph input: ' + Model.Inputs[I].Name);
    WriteLn('Compiling ', UTF8Encode(Name));
    if Quantize then
    begin
      Lowered := QuantizeTextGraph(Model, Stage, Spec, Written);
      CompileWindowsTvmOnnxLibrary(Lowered, Kit, LibraryFile, NvrtcLibrary, Architecture);
    end
    else
      CompileWindowsTvmOnnxLibrary(Model, Kit, LibraryFile, NvrtcLibrary, Architecture);
  finally
    Lowered.Free;
    Model.Free;
  end;
  Artifacts.AddPair(Name, FileHash(LibraryFile));
end;

procedure SaveManifest(const FileName: string; const Manifest: TJSONObject);
var
  Stream: TFileStream;
  Raw: UTF8String;
begin
  Raw := UTF8Encode(Manifest.ToJSON + #10);
  Stream := TFileStream.Create(FileName, fmCreate or fmShareExclusive);
  try
    Stream.WriteBuffer(Raw[1], Length(Raw));
  finally
    Stream.Free;
  end;
end;

procedure RemoveStage(const Folder: string);
var
  Item: TSearchRec;
  Status: Integer;
begin
  Status := FindFirst(IncludeTrailingPathDelimiter(Folder) + '*', faAnyFile, Item);
  if Status = 0 then
    try
      while Status = 0 do
      begin
        if (Item.Attr and faDirectory) = 0 then DeleteFileW(PWideChar(FileIn(Folder, Item.Name)));
        Status := FindNext(Item);
      end;
    finally
      SysUtils.FindClose(Item);
    end;
  RemoveDir(Folder);
end;

procedure KeepUsedWeights(const Stage: string; const Graphs, Weights: TJSONObject;
  const Written: TStringList);
var
  Used: TStringList;
  Params, Spec: TJSONObject;
  I, J, At: Integer;
  Name, FileName: string;
begin
  Used := TStringList.Create;
  try
    Used.Sorted := True;
    Used.Duplicates := dupIgnore;
    Spec := ObjectValue(Weights.GetValue('model.language_model.embed_tokens.weight'), 'embedding table');
    Used.Add(StringValue(Spec, 'file'));
    for I := 0 to Graphs.Count - 1 do
    begin
      Spec := ObjectValue(JSONObjectValue(Graphs, I), 'graph');
      Params := ObjectValue(Spec.GetValue('parameters'), 'graph parameters');
      for J := 0 to Params.Count - 1 do
        Used.Add(StringValue(ObjectValue(JSONObjectValue(Params, J), 'weight'), 'file'));
    end;
    for I := Weights.Count - 1 downto 0 do
    begin
      Name := JSONObjectName(Weights, I);
      FileName := StringValue(ObjectValue(JSONObjectValue(Weights, I), Name), 'file');
      if Used.IndexOf(FileName) >= 0 then Continue;
      At := Written.IndexOfName(FileName);
      if At >= 0 then
      begin
        Check(DeleteFileW(PWideChar(FileIn(Stage, FileName))), 'Cannot remove unused weight: ' + FileName);
        Written.Delete(At);
      end;
      DeleteField(Weights, Name);
    end;
  finally
    Used.Free;
  end;
end;

procedure Convert(const Source, Profile, Destination: string; const Kit: TWindowsTvmKit;
  const NvrtcLibrary, Architecture: string);
var
  Checkpoint: TSafeTensorCheckpoint;
  Manifest, Parameters, Weights, Sources, Artifacts, Graphs: TJSONObject;
  Written, SourceNames: TStringList;
  Stage, Name, Origin: string;
  ID: TGUID;
  I, WeightCount: Integer;
  Staging: Boolean;
begin
  Check(DirectoryExists(Source) and DirectoryExists(Profile), 'Source and profile directories must exist');
  Check(DirectoryExists(ExtractFileDir(Destination)) and not DirectoryExists(Destination) and
    not FileExists(Destination), 'Choose a new destination in an existing directory');
  Check(CreateGUID(ID) = 0, 'Cannot create a staging identifier');
  Stage := Destination + '.partial-' + GUIDToString(ID);
  Staging := False;
  Manifest := ReadObject(FileIn(Profile, 'model.json'));
  try
    Check((StringValue(Manifest, 'schema') = 'crosspascal.qwen2_5_vl.v1') and
      (StringValue(Manifest, 'backend') = 'TVM LLVM CPU FP32') and
      (StringValue(Manifest, 'cpu_arch') = 'haswell') and
      (JSONValueText(Manifest.GetValue('image_size')) = '224'),
      'Expected the Qwen2.5-VL 224-pixel Haswell CPU profile');
    Artifacts := ObjectValue(Manifest.GetValue('artifact_sha256'), 'artifact hashes');
    CheckHash(FileIn(Profile, 'parameters.json'), StringValue(Artifacts, 'parameters.json'));
    Parameters := ReadObject(FileIn(Profile, 'parameters.json'));
    try
      Sources := ObjectValue(Parameters.GetValue('source_sha256'), 'original source hashes');
      Weights := ObjectValue(Parameters.GetValue('weights'), 'weights');
      Check(Sources.Count > 0, 'The profile has no original checkpoint hashes');
      Checkpoint := TSafeTensorCheckpoint.Create(Source);
      try
        WriteLn('Verifying the original checkpoint');
        for I := 0 to Sources.Count - 1 do
        begin
          Name := JSONObjectName(Sources, I);
          CheckHash(FileIn(Source, Name), StringValue(Sources, Name));
        end;
        SourceNames := TStringList.Create;
        Written := TStringList.Create;
        try
          SourceNames.CaseSensitive := True;
          for Name in Checkpoint.TensorNames do SourceNames.Add(Name);
          SourceNames.Sort;
          Check(CreateDir(Stage), 'Cannot create staging directory');
          Staging := True;
          for I := 0 to Artifacts.Count - 1 do
          begin
            Name := JSONObjectName(Artifacts, I);
            if SameText(ExtractFileExt(Name), '.dll') then Continue;
            Origin := FileIn(Profile, Name);
            if Sources.GetValue(Name) <> nil then Origin := FileIn(Source, Name);
            CopyChecked(Origin, FileIn(Stage, Name), StringValue(Artifacts, Name));
          end;
          Graphs := ObjectValue(Manifest.GetValue('graphs'), 'graphs');
          Check(Graphs.Count = 3, 'Expected text, vision and prefill graphs');
          ExportWeights(Checkpoint, Weights, Stage, Written, SourceNames);
          CopyGraphBuffers(Manifest, Profile, Stage, Written);
          CompileGraph(Stage, ObjectValue(Graphs.GetValue('vision'), 'vision'), Artifacts,
            'qwen-vl-vision-cpu.dll', Kit, NvrtcLibrary, Architecture, False, Written);
          CompileGraph(Stage, ObjectValue(Graphs.GetValue('text'), 'text'), Artifacts,
            'qwen-vl-text-cpu.dll', Kit, NvrtcLibrary, Architecture, Architecture <> '', Written);
          CompileGraph(Stage, ObjectValue(Graphs.GetValue('prefill'), 'prefill'), Artifacts,
            'qwen-vl-prefill-cpu.dll', Kit, NvrtcLibrary, Architecture, Architecture <> '', Written);
          if FileExists(FileIn(Source, 'LICENSE')) then
            CopyChecked(FileIn(Source, 'LICENSE'), FileIn(Stage, 'LICENSE'), StringValue(Sources, 'LICENSE'));
          DeleteField(Manifest, 'inference_acceptance');
          DeleteField(Manifest, 'delivery_acceptance');
          DeleteField(Manifest, 'compiler');
          if Architecture = '' then
            Manifest.AddPair('compiler', 'TVM native LLVM Haswell')
          else begin
            KeepUsedWeights(Stage, Graphs, Weights, Written);
            SaveManifest(FileIn(Stage, 'parameters.json'), Parameters);
            DeleteField(Artifacts, 'parameters.json');
            Artifacts.AddPair('parameters.json', FileHash(FileIn(Stage, 'parameters.json')));
            DeleteField(Manifest, 'backend');
            DeleteField(Manifest, 'cpu_arch');
            Manifest.AddPair('backend', 'TVM CUDA W8A32');
            Manifest.AddPair('cuda_arch', Architecture);
            Manifest.AddPair('compiler', 'TVM native CUDA');
            Manifest.AddPair('vision_precision', 'FP32');
          end;
          WeightCount := Written.Count;
          SaveManifest(FileIn(Stage, 'model.json'), Manifest);
          Check(MoveFileW(PWideChar(Stage), PWideChar(Destination)),
            'Cannot move the verified bundle to its destination');
          Staging := False;
          WriteLn('READY: ', WeightCount, ' weight files generated from the original checkpoint');
        finally
          Written.Free;
          SourceNames.Free;
        end;
      finally
        Checkpoint.Free;
      end;
    finally
      Parameters.Free;
    end;
  finally
    Manifest.Free;
    if Staging then RemoveStage(Stage);
  end;
end;

var
  Kit: TWindowsTvmKit;
  NvrtcLibrary, Architecture: string;

begin
  try
    if (ParamCount <> 9) and not ((ParamCount = 12) and (ParamStr(10) = '--cuda')) then
    begin
      WriteLn(ErrOutput, 'Usage: ConvertQwenVLProfile original-directory profile-directory output-directory');
      WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
      WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_XX]');
      Halt(2);
    end;
    Kit.CompilerDirectory := ParamStr(4);
    Kit.Linker := ParamStr(5);
    Kit.FfiLibrary := ExpandFileName(ParamStr(6));
    Kit.MsvcLibraries := ParamStr(7);
    Kit.UcrtLibraries := ParamStr(8);
    Kit.WindowsLibraries := ParamStr(9);
    if ParamCount = 12 then
    begin
      NvrtcLibrary := ParamStr(11);
      Architecture := ParamStr(12);
    end;
    Convert(ExpandFileName(ParamStr(1)), ExpandFileName(ParamStr(2)), ExpandFileName(ParamStr(3)),
      Kit, NvrtcLibrary, Architecture);
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, E.Message);
      Halt(1);
    end;
  end;
end.
