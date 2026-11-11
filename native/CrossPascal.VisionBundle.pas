{ ************************************************************************** }
{                                                                            }
{ CrossPascal.VisionBundle                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.VisionBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.TVMCompiler;

procedure ConvertCpuVisionBundle(const SourceFile, CatalogFile, ProfileID, Destination: string;
  const Kit: TWindowsTvmKit; const RecipeFile: string = '');
procedure ConvertVisionBundle(const SourceFile, CatalogFile, ProfileID, Destination: string;
  const Kit: TWindowsTvmKit; const RecipeFile, NvrtcLibrary, Architecture: string);

implementation

uses
  {$IFDEF FPC}SysUtils, Classes, Windows{$ELSE}System.SysUtils, System.Classes, Winapi.Windows{$ENDIF},
  CrossPascal.Json, CrossPascal.NativeBundle, CrossPascal.Onnx, CrossPascal.SHA256,
  CrossPascal.RtDetrSource, CrossPascal.TVMOnnx, CrossPascal.YoloSource, TVM.CpuGraph;

function BuildTimeUTC: string;
var
  Time: TSystemTime;
begin
  GetSystemTime(Time);
  Result := Format('%.4d-%.2d-%.2dT%.2d:%.2d:%.2dZ',
    [Time.wYear, Time.wMonth, Time.wDay, Time.wHour, Time.wMinute, Time.wSecond]);
end;

procedure WriteUTF8(const FileName, Value: string);
var
  Stream: TFileStream;
  Encoded: UTF8String;
begin
  Encoded := UTF8Encode(Value);
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    if Length(Encoded) > 0 then Stream.WriteBuffer(Encoded[1], Length(Encoded));
  finally
    Stream.Free;
  end;
end;

function InputSidecar(const Model: TOnnxModel; const IOProfile: TJSONObject): string;
var
  Entry: TOnnxValue;
  Constant: TOnnxTensor;
  Shape: TArray<Int64>;
  Name, Dimensions: string;
  Count, I: Integer;
  IsInitializer: Boolean;
begin
  Name := ModelText(IOProfile.GetValue('name'));
  Shape := ModelShape(IOProfile.GetValue('shape'));
  RequireCpu((Name <> '') and (Pos(#9, Name) = 0) and (Pos(#10, Name) = 0) and (Pos(#13, Name) = 0),
    'Invalid vision input name');
  RequireCpu((Length(Shape) = 4) and (Shape[0] = 1) and (Shape[1] = 3) and
    (Shape[2] = Shape[3]) and (ModelText(IOProfile.GetValue('layout')) = 'NCHW') and
    (ModelText(IOProfile.GetValue('dtype')) = 'float32'), 'Expected a batch-one FP32 image input');
  Count := 0;
  for Entry in Model.Inputs do
  begin
    IsInitializer := False;
    for Constant in Model.Initializers do
      if Entry.Name = Constant.Name then
      begin
        IsInitializer := True;
        Break;
      end;
    if IsInitializer then Continue;
    Inc(Count);
    RequireCpu((Entry.Name = Name) and Entry.HasShape and (Entry.DataType = 1) and
      (Length(Entry.Shape) = Length(Shape)), 'The ONNX input differs from the vision profile');
    for I := 0 to High(Shape) do
      RequireCpu((Entry.Shape[I].Value = Shape[I]) or ((I = 0) and (Entry.Shape[I].Value < 0)),
        'The ONNX input dimensions differ from the vision profile');
  end;
  RequireCpu(Count = 1, 'The vision bundle requires a single image input');
  Dimensions := IntToStr(Shape[0]);
  for I := 1 to High(Shape) do Dimensions := Dimensions + ',' + IntToStr(Shape[I]);
  Result := '0'#9 + Name + #9 + Dimensions + sLineBreak;
end;

function VisionManifest(const Catalog: TJSONValue;
  const ProfileID, Source, SourceHash, RecipeHash, BindingsHash, ModelFile, Architecture: string): TJSONObject;
var
  Root, Profile, Artifact, IO, Provenance, Parameters, Toolchain: TJSONObject;
  Models, Digests, Shape, Mean, Std, Numbers: TJSONArray;
  Entry: TJSONObject;
  I, Size: Integer;
  Pinned: Boolean;
  Task, Adapter: string;
begin
  Root := ModelObject(Catalog, 'vision profile catalog');
  RequireCpu(ModelText(Root.GetValue('schema')) = 'crosspascal.cpu-vision-profiles.v1',
    'Unsupported CPU vision profile catalog');
  Models := ModelArray(Root.GetValue('models'), 'vision profiles');
  Profile := nil;
  for I := 0 to Models.Count - 1 do
  begin
    Entry := ModelObject(Models.Items[I], 'vision profile');
    if ModelText(Entry.GetValue('id')) = ProfileID then
    begin
      RequireCpu(Profile = nil, 'Duplicate vision profile: ' + ProfileID);
      Profile := Entry;
    end;
  end;
  RequireCpu(Profile <> nil, 'Unknown vision profile: ' + ProfileID);
  Digests := ModelArray(Profile.GetValue('accepted_source_sha256'), 'vision source hashes');
  Pinned := False;
  for I := 0 to Digests.Count - 1 do
    if SameText(ModelText(Digests.Items[I]), SourceHash) then Pinned := True;
  RequireCpu(Pinned, 'The checkpoint does not match the selected profile');
  if RecipeHash <> '' then
    RequireCpu(SameText(RecipeHash, ModelText(Profile.GetValue('graph_recipe_sha256'))),
      'The graph recipe does not match the selected profile');
  if BindingsHash <> '' then
    RequireCpu(SameText(BindingsHash, ModelText(Profile.GetValue('graph_bindings_sha256'))),
      'The graph bindings do not match the selected profile');
  Task := ModelText(Profile.GetValue('task_kind'));
  Adapter := ModelText(Profile.GetValue('adapter'));
  RequireCpu(((Task = 'classification') and (Adapter = 'yolo11')) or
    ((Task = 'object_detection') and ((Adapter = 'yolo11') or (Adapter = 'rtdetr'))),
    'Unsupported CPU vision task or adapter');
  Size := ModelInt(Profile.GetValue('input_size'), 'vision input size');
  RequireCpu((Size > 0) and (Size <= 4096), 'Invalid CPU vision input size');
  Result := TJSONObject.Create;
  try
    Result.AddPair('format_version', JSONInteger(1));
    Result.AddPair('task_kind', Task);
    Result.AddPair('adapter', Adapter);
    Result.AddPair('adapter_api_version', JSONInteger(1));
    Result.AddPair('license', ModelText(Profile.GetValue('source_license')));
    Artifact := TJSONObject.Create;
    Result.AddPair('artifact', Artifact);
    Artifact.AddPair('os', 'windows');
    Artifact.AddPair('arch', 'x86_64');
    if Architecture = '' then Artifact.AddPair('device', 'cpu')
    else
    begin
      Artifact.AddPair('device', 'cuda');
      Artifact.AddPair('cuda_arch', Architecture);
    end;
    Artifact.AddPair('dtype', 'float32');
    Artifact.AddPair('file', ModelFile);
    Artifact.AddPair('entry', 'main');
    Artifact.AddPair('format', 'so');
    IO := TJSONObject.Create;
    Result.AddPair('io', IO);
    IO.AddPair('name', ModelText(Profile.GetValue('input_name')));
    IO.AddPair('layout', 'NCHW');
    IO.AddPair('dtype', 'float32');
    Shape := TJSONArray.Create;
    IO.AddPair('shape', Shape);
    Shape.AddElement(JSONInteger(1));
    Shape.AddElement(JSONInteger(3));
    Shape.AddElement(JSONInteger(Size));
    Shape.AddElement(JSONInteger(Size));
    if Task = 'classification' then
    begin
      Mean := ModelArray(Profile.GetValue('mean'), 'vision normalization mean');
      Std := ModelArray(Profile.GetValue('std'), 'vision normalization standard deviation');
      RequireCpu((Mean.Count = Std.Count) and (Mean.Count in [0, 3]), 'Invalid vision normalization');
      if Mean.Count > 0 then
      begin
        Parameters := TJSONObject.Create;
        Result.AddPair('vision', Parameters);
        Numbers := TJSONArray.Create;
        Parameters.AddPair('mean', Numbers);
        for I := 0 to Mean.Count - 1 do
          Numbers.AddElement(JSONFloat(ModelFloat(Mean.Items[I], 'mean')));
        Numbers := TJSONArray.Create;
        Parameters.AddPair('std', Numbers);
        for I := 0 to Std.Count - 1 do
          Numbers.AddElement(JSONFloat(ModelFloat(Std.Items[I], 'std')));
      end;
    end
    else begin
      Parameters := TJSONObject.Create;
      Result.AddPair('defaults', Parameters);
      Parameters.AddPair('score_threshold',
        JSONFloat(ModelFloat(Profile.GetValue('score_threshold'), 'score threshold')));
      if Adapter = 'yolo11' then
        Parameters.AddPair('nms_threshold',
          JSONFloat(ModelFloat(Profile.GetValue('nms_threshold'), 'NMS threshold')));
    end;
    Provenance := TJSONObject.Create;
    Result.AddPair('provenance', Provenance);
    Provenance.AddPair('source', Source);
    Provenance.AddPair('source_hash', SourceHash);
    Provenance.AddPair('source_url', ModelText(Profile.GetValue('source_url')));
    if RecipeHash <> '' then Provenance.AddPair('graph_recipe_hash', RecipeHash);
    if BindingsHash <> '' then Provenance.AddPair('graph_bindings_hash', BindingsHash);
    Provenance.AddPair('built_at', BuildTimeUTC);
    Toolchain := TJSONObject.Create;
    Result.AddPair('toolchain', Toolchain);
    Toolchain.AddPair('tvm', '0.26.0');
    Toolchain.AddPair('frontend', 'CrossPascal ONNX');
  except
    Result.Free;
    raise;
  end;
end;

procedure ConvertVisionBundle(const SourceFile, CatalogFile, ProfileID, Destination: string;
  const Kit: TWindowsTvmKit; const RecipeFile, NvrtcLibrary, Architecture: string);
var
  Source, RecipeSource, BindingSource: TFileStream;
  Catalog, BindingData: TJSONValue;
  Root, Artifact: TJSONObject;
  Graph, Recipe: TOnnxModel;
  SourcePath, Output, Stage, ModelPath, ModelFile, Sidecar, RecipeHash, BindingsHash, BindingsFile: string;
  ID: TGUID;
  IsTorch, IsSafeTensors: Boolean;
begin
  if Architecture = '' then ModelFile := 'model.cpu.dll'
  else ModelFile := 'model.cuda.dll';
  SourcePath := ExpandFileName(SourceFile);
  Output := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireCpu((not DirectoryExists(Output)) and (not FileExists(Output)) and
    DirectoryExists(ExtractFileDir(Output)), 'Choose a new bundle directory in an existing parent');
  Source := TFileStream.Create(SourcePath, fmOpenRead or fmShareDenyWrite);
  Catalog := nil;
  Root := nil;
  RecipeSource := nil;
  BindingSource := nil;
  BindingData := nil;
  Graph := nil;
  Recipe := nil;
  Stage := '';
  try
    IsTorch := SameText(ExtractFileExt(SourcePath), '.pt');
    IsSafeTensors := SameText(ExtractFileExt(SourcePath), '.safetensors');
    RecipeHash := '';
    BindingsHash := '';
    if IsTorch or IsSafeTensors then
    begin
      RequireCpu((RecipeFile <> '') and
        ((IsTorch and ((ProfileID = 'yolo11n') or (ProfileID = 'yolo11n-cls'))) or
        (IsSafeTensors and (ProfileID = 'rtdetr-r18vd'))),
        'Select the graph recipe for the original checkpoint');
      RecipeSource := TFileStream.Create(RecipeFile, fmOpenRead or fmShareDenyWrite);
      RecipeHash := SHA256Stream(RecipeSource);
      if IsSafeTensors then
      begin
        BindingsFile := BundleMember(ExtractFileDir(ExpandFileName(RecipeFile)), 'bindings.json');
        BindingSource := TFileStream.Create(BindingsFile, fmOpenRead or fmShareDenyWrite);
        BindingsHash := SHA256Stream(BindingSource);
      end;
    end
    else
      RequireCpu(SameText(ExtractFileExt(SourcePath), '.onnx'),
        'Expected an ONNX model or a supported checkpoint');
    Catalog := ReadModelJSON(CatalogFile);
    Root := VisionManifest(Catalog, ProfileID, ExtractFileName(SourcePath), SHA256Stream(Source),
      RecipeHash, BindingsHash, ModelFile, Architecture);
    if IsTorch or IsSafeTensors then
    begin
      Recipe := TOnnxModel.Create(RecipeFile);
      if IsTorch then
        Graph := BindYoloCheckpoint(Recipe, SourcePath)
      else begin
        BindingData := ReadModelJSON(BindingsFile);
        Graph := BindRtDetrCheckpoint(Recipe, SourcePath, ModelObject(BindingData, 'RT-DETR bindings'));
      end;
    end
    else
      Graph := TOnnxModel.Create(SourcePath);
    Artifact := ModelObject(Root.GetValue('artifact'), 'vision artifact');
    Sidecar := InputSidecar(Graph, ModelObject(Root.GetValue('io'), 'vision inputs'));
    RequireCpu(CreateGUID(ID) = 0, 'Could not create a staging directory name');
    ModelPath := Output + '.building-' + GUIDToString(ID);
    RequireCpu(CreateDir(ModelPath), 'Could not create the vision build directory');
    Stage := ModelPath;
    ModelPath := BundleMember(Stage, ModelFile);
    CompileWindowsTvmOnnxLibrary(Graph, Kit, ModelPath, NvrtcLibrary, Architecture);
    Artifact.AddPair('hash', BundleFileSHA256(ModelPath));
    WriteUTF8(BundleMember(Stage, ModelFile + '.inputs'), Sidecar);
    WriteBundleJSON(BundleMember(Stage, 'manifest.json'), Root);
    RequireCpu(MoveFileW(PWideChar(Stage), PWideChar(Output)), 'Could not install the new vision bundle');
    Stage := '';
  finally
    if Stage <> '' then DeleteBundleStage(Stage);
    Graph.Free;
    Recipe.Free;
    BindingData.Free;
    BindingSource.Free;
    RecipeSource.Free;
    Root.Free;
    Catalog.Free;
    Source.Free;
  end;
end;

procedure ConvertCpuVisionBundle(const SourceFile, CatalogFile, ProfileID, Destination: string;
  const Kit: TWindowsTvmKit; const RecipeFile: string);
begin
  ConvertVisionBundle(SourceFile, CatalogFile, ProfileID, Destination, Kit, RecipeFile, '', '');
end;

end.
