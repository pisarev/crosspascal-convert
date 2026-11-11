{ ************************************************************************** }
{                                                                            }
{ ConvertQwenVLProfile                                                       }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

program ConvertQwenVLProfile;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils, Classes, Windows{$ELSE}System.SysUtils, System.Classes, Winapi.Windows{$ENDIF},
  CrossPascal.Json, CrossPascal.Onnx, CrossPascal.SafeTensors, CrossPascal.SHA256,
  CrossPascal.TVMCompiler, CrossPascal.TVMOnnx;

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

procedure CompileGraph(const Stage: string; const Spec, Artifacts: TJSONObject;
  const ExpectedLibrary: string; const Kit: TWindowsTvmKit);
var
  Model: TOnnxModel;
  Name, LibraryFile: string;
  Inputs: TJSONValue;
  I: Integer;
begin
  Name := StringValue(Spec, 'library');
  Check(Name = ExpectedLibrary, 'Unexpected Qwen-VL library: ' + Name);
  LibraryFile := FileIn(Stage, Name);
  Model := TOnnxModel.Create(FileIn(Stage, StringValue(Spec, 'onnx')));
  try
    Inputs := Spec.GetValue('input_names');
    Check((Inputs is TJSONArray) and (TJSONArray(Inputs).Count = Length(Model.Inputs)),
      'Qwen-VL graph inputs differ from its profile');
    for I := 0 to High(Model.Inputs) do
      Check(Model.Inputs[I].Name = JSONValueText(TJSONArray(Inputs).Items[I]),
        'Unexpected Qwen-VL graph input: ' + Model.Inputs[I].Name);
    WriteLn('Compiling ', UTF8Encode(Name));
    CompileWindowsTvmOnnxLibrary(Model, Kit, LibraryFile);
  finally
    Model.Free;
  end;
  DeleteField(Artifacts, Name);
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

procedure Convert(const Source, Profile, Destination: string; const Kit: TWindowsTvmKit);
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
          CompileGraph(Stage, ObjectValue(Graphs.GetValue('vision'), 'vision'), Artifacts,
            'qwen-vl-vision-cpu.dll', Kit);
          CompileGraph(Stage, ObjectValue(Graphs.GetValue('text'), 'text'), Artifacts,
            'qwen-vl-text-cpu.dll', Kit);
          CompileGraph(Stage, ObjectValue(Graphs.GetValue('prefill'), 'prefill'), Artifacts,
            'qwen-vl-prefill-cpu.dll', Kit);
          ExportWeights(Checkpoint, Weights, Stage, Written, SourceNames);
          WeightCount := Written.Count;
          CopyGraphBuffers(Manifest, Profile, Stage, Written);
          if FileExists(FileIn(Source, 'LICENSE')) then
            CopyChecked(FileIn(Source, 'LICENSE'), FileIn(Stage, 'LICENSE'), StringValue(Sources, 'LICENSE'));
          DeleteField(Manifest, 'inference_acceptance');
          DeleteField(Manifest, 'delivery_acceptance');
          DeleteField(Manifest, 'compiler');
          Manifest.AddPair('compiler', 'TVM native LLVM Haswell');
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

begin
  try
    if ParamCount <> 9 then
    begin
      WriteLn(ErrOutput, 'Usage: ConvertQwenVLProfile original-directory profile-directory output-directory');
      WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
      Halt(2);
    end;
    Kit.CompilerDirectory := ParamStr(4);
    Kit.Linker := ParamStr(5);
    Kit.FfiLibrary := ExpandFileName(ParamStr(6));
    Kit.MsvcLibraries := ParamStr(7);
    Kit.UcrtLibraries := ParamStr(8);
    Kit.WindowsLibraries := ParamStr(9);
    Convert(ExpandFileName(ParamStr(1)), ExpandFileName(ParamStr(2)), ExpandFileName(ParamStr(3)), Kit);
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, E.Message);
      Halt(1);
    end;
  end;
end.
