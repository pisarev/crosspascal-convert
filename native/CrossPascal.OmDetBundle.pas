{ ************************************************************************** }
{                                                                            }
{ CrossPascal.OmDetBundle                                                     }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.OmDetBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.TVMCompiler;

procedure ConvertOmDetBundle(const SourceDirectory, Destination: string; const Kit: TWindowsTvmKit;
  const NvrtcLibrary, Architecture: string);

implementation

uses
  {$IFDEF FPC}Windows, SysUtils, Classes{$ELSE}Winapi.Windows, System.SysUtils, System.Classes{$ENDIF},
  CrossPascal.Json, CrossPascal.NativeBundle, CrossPascal.Onnx, CrossPascal.SHA256,
  CrossPascal.TVMOnnx, TVM.CpuGraph, CrossPascal.NativeIO;

const
  CheckpointHash = '439d1575d7e237ad565ed6969ea2a2dfcedf2086155e9ca3ac96cd6180a48cfd';
  SourceNames: array[0..4] of string = ('text.onnx', 'detector.onnx',
    'class-projection.f32', 'vocab.json', 'merges.txt');
  SourceHashes: array[0..4] of string = (
    '65c63f2f54cae6d407641038a968cdf485ea97b70514097ac1efbb0665fb8a04',
    '016a4ae886e0cec55503692cd75a50c8f9f0337de90885d5165584a61ca3e1e5',
    'b52b945b89ddbae183f89564e479dbc46d683ade99a3bce151140e67b815275d',
    'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
    '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a');
  SourceSizes: array[0..4] of Int64 = (252812393, 215920478, 1048576, 1059962, 524619);
  Roles: array[0..4] of string = ('text', 'detector', 'projection', 'vocab', 'merges');

function BundleManifest(const Device, Architecture: string): TJSONObject;
var
  Provenance: TJSONObject;
begin
  Result := TJSONObject.Create;
  try
    Result.AddPair('format_version', JSONInteger(1));
    Result.AddPair('adapter', 'omdet-turbo');
    Result.AddPair('task_kind', 'text_guided_detection');
    Result.AddPair('profile', 'omdet-turbo-tiny16');
    Result.AddPair('os', 'windows');
    Result.AddPair('arch', 'x86_64');
    Result.AddPair('dtype', 'float32');
    Result.AddPair('cpu_target', 'haswell');
    Result.AddPair('device', Device);
    Result.AddPair('cuda_arch', Architecture);
    Result.AddPair('image_side', JSONInteger(640));
    Result.AddPair('class_capacity', JSONInteger(16));
    Result.AddPair('token_capacity', JSONInteger(77));
    Result.AddPair('hidden_size', JSONInteger(512));
    Result.AddPair('query_count', JSONInteger(900));
    Result.AddPair('preprocessing', 'rgb-pillow-bilinear-omdet');
    Result.AddPair('postprocessing', 'sigmoid-valid-prefix-topk900-class-nms-clip');
    Result.AddPair('license', 'Apache-2.0');
    Provenance := TJSONObject.Create;
    Result.AddPair('provenance', Provenance);
    Provenance.AddPair('source', 'omlab/omdet-turbo-swin-tiny-hf');
    Provenance.AddPair('revision', '7fe93cecfb770c4d76cf71163956221249cab566');
    Provenance.AddPair('checkpoint_sha256', CheckpointHash);
    Provenance.AddPair('text_onnx_sha256', SourceHashes[0]);
    Provenance.AddPair('detector_onnx_sha256', SourceHashes[1]);
    Result.AddPair('artifacts', TJSONArray.Create);
  except
    Result.Free;
    raise;
  end;
end;

procedure ConvertOmDetBundle(const SourceDirectory, Destination: string; const Kit: TWindowsTvmKit;
  const NvrtcLibrary, Architecture: string);
var
  Sources: array[0..4] of TStream;
  Root, Artifact: TJSONObject;
  Artifacts: TJSONArray;
  Graph: TOnnxModel;
  Target: TStream;
  Folder, Output, Stage, Device, Name: string;
  ID: TGUID;
  I: Integer;
begin
  Folder := ExcludeTrailingPathDelimiter(ExpandFileName(SourceDirectory));
  Output := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireCpu(DirectoryExists(Folder), 'Select the OmDet ONNX source-pack directory');
  RequireCpu((not DirectoryExists(Output)) and (not FileExists(Output)) and
    DirectoryExists(ExtractFileDir(Output)), 'Choose a new bundle directory in an existing parent');
  RequireCpu((Architecture = '') = (NvrtcLibrary = ''), 'Supply both the CUDA target and NVRTC library');
  if Architecture = '' then
    Device := 'cpu'
  else
    Device := 'cuda';
  for I := 0 to High(Sources) do Sources[I] := nil;
  Root := nil;
  Graph := nil;
  Stage := '';
  try
    { Keep the exact trained source files read-locked through compilation. }
    for I := 0 to High(Sources) do
    begin
      Sources[I] := TNativeReadStream.Create(BundleMember(Folder, SourceNames[I]));
      RequireCpu(Sources[I].Size = SourceSizes[I], 'OmDet source size differs: ' + SourceNames[I]);
      RequireCpu(SameText(SHA256Stream(Sources[I]), SourceHashes[I]),
        'OmDet source SHA-256 differs: ' + SourceNames[I]);
      Sources[I].Position := 0;
    end;
    Root := BundleManifest(Device, Architecture);
    Artifacts := ModelArray(Root.GetValue('artifacts'), 'OmDet artifacts');
    RequireCpu(CreateGUID(ID) = 0, 'Could not create an OmDet staging name');
    Name := Output + '.building-' + GUIDToString(ID);
    RequireCpu(CreateDir(Name), 'Could not create the OmDet build directory');
    Stage := Name;
    for I := 0 to High(Sources) do
    begin
      if I < 2 then
      begin
        Name := Roles[I] + '.' + Device + '.dll';
        Graph := TOnnxModel.Create(BundleMember(Folder, SourceNames[I]));
        try
          if I = 0 then
            Graph.SpecializeInput('input_ids', [1, 77])
          else
            Graph.SpecializeInput('pixel_values', [1, 3, 640, 640]);
          CompileWindowsTvmOnnxLibrary(Graph, Kit, BundleMember(Stage, Name), NvrtcLibrary, Architecture);
        finally
          FreeAndNil(Graph);
        end;
      end
      else begin
        Name := SourceNames[I];
        Target := TNativeWriteStream.Create(BundleMember(Stage, Name), False);
        try
          Target.CopyFrom(Sources[I], Sources[I].Size);
        finally
          Target.Free;
        end;
      end;
      Artifact := TJSONObject.Create;
      Artifacts.AddElement(Artifact);
      Artifact.AddPair('role', Roles[I]);
      Artifact.AddPair('file', Name);
      Artifact.AddPair('sha256', BundleFileSHA256(BundleMember(Stage, Name)));
      Target := TNativeReadStream.Create(BundleMember(Stage, Name));
      try
        Artifact.AddPair('bytes', JSONInteger(Target.Size));
      finally
        Target.Free;
      end;
    end;
    WriteBundleJSON(BundleMember(Stage, 'manifest.json'), Root);
    RequireCpu(MoveFileW(PWideChar(Stage), PWideChar(Output)), 'Could not install the new OmDet bundle');
    Stage := '';
  finally
    if Stage <> '' then DeleteBundleStage(Stage);
    Graph.Free;
    Root.Free;
    for I := High(Sources) downto 0 do Sources[I].Free;
  end;
end;

end.
