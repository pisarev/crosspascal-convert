{ ************************************************************************** }
{                                                                            }
{ ConvertQwen35                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

program ConvertQwen35;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}Windows, SysUtils{$ELSE}Winapi.Windows, System.SysUtils{$ENDIF}, CrossPascal.Json,
  CrossPascal.Qwen35Source, CrossPascal.Qwen35Package, TVM.CpuGraph, CrossPascal.NativeIO;

type
  TLogger = class
    procedure Progress(const Text: string);
  end;

procedure TLogger.Progress(const Text: string);
begin
  WriteLn(UTF8Encode(Text));
  Flush(Output);
end;

{ Copies the optional image encoder fields of a launcher source identity. Only these seven
  names are taken; InspectQwen35TextSource validates their content. Relative paths resolve
  beside the identity JSON. }
procedure AddIdentityVision(const Identity, Root: TJSONObject; const Folder: string);
const
  Fields: array[0..6] of string = ('vision_source', 'vision_config', 'vision_license_file',
    'vision_source_repository', 'vision_source_revision', 'vision_height', 'vision_width');
var
  Value: TJSONValue;
  Item: string;
  I, J: Integer;
  Known: Boolean;
begin
  for I := 0 to Identity.Count - 1 do
  begin
    Item := JSONObjectName(Identity, I);
    if Copy(Item, 1, 7) <> 'vision_' then Continue;
    Known := False;
    for J := 0 to High(Fields) do
      Known := Known or (Item = Fields[J]);
    RequireCpu(Known, 'Unsupported source identity field: ' + Item);
  end;
  for I := 0 to High(Fields) do
  begin
    Value := Identity.GetValue(Fields[I]);
    if Value = nil then Continue;
    if I >= 5 then
      Root.AddPair(Fields[I], JSONInteger(ModelInt(Value, Fields[I])))
    else begin
      RequireCpu((Value is TJSONString) and not (Value is TJSONNumber),
        'Source identity ' + Fields[I] + ' must be a JSON string');
      Item := JSONValueText(Value);
      if (I <= 2) and (Item <> '') then
      begin
        if ExtractFileDrive(Item) = '' then Item := Folder + Item;
        Item := ExpandFileName(Item);
      end;
      Root.AddPair(Fields[I], Item);
    end;
  end;
end;

function LauncherRequest: TJSONValue;
const
  ReaderLicenseNames: array[0..2] of string = ('LICENSE-GGML.txt',
    'LICENSE-GGML-vendors.txt', 'LICENSE-LLVM.txt');
  ReaderProvenanceNames: array[0..1] of string = ('SOURCE-GGML-reader.txt',
    'LICENSE-OLLAMA.txt');
var
  IdentityData: TJSONValue;
  Identity, Root, Kit, Licenses: TJSONObject;
  LicenseFile, Name, IdentityFolder: string;
begin
  RequireCpu(NativeParamCount = 17, 'Invalid native launcher arguments');
  IdentityData := ReadModelJSON(NativeParamStr(4));
  Root := TJSONObject.Create;
  try
    Identity := ModelObject(IdentityData, 'source identity');
    IdentityFolder := ExtractFilePath(ExpandFileName(NativeParamStr(4)));
    Root.AddPair('source', NativeParamStr(2));
    Root.AddPair('destination', NativeParamStr(3));
    Root.AddPair('context', JSONInteger(StrToInt(NativeParamStr(5))));
    Root.AddPair('head_rows', JSONInteger(4096));
    Root.AddPair('target', NativeParamStr(6));
    Root.AddPair('cuda_arch', NativeParamStr(7));
    Root.AddPair('ggml', NativeParamStr(8));
    Root.AddPair('packed_kernel_directory', NativeParamStr(9));
    Root.AddPair('source_repository', ModelText(Identity.GetValue('source_repository')));
    Root.AddPair('source_revision', ModelText(Identity.GetValue('source_revision')));
    LicenseFile := ModelText(Identity.GetValue('license_file'));
    RequireCpu(LicenseFile <> '', 'Source identity must specify license_file');
    if ExtractFileDrive(LicenseFile) = '' then
      LicenseFile := IdentityFolder + LicenseFile;
    Root.AddPair('license_file', ExpandFileName(LicenseFile));
    AddIdentityVision(Identity, Root, IdentityFolder);
    Licenses := TJSONObject.Create;
    Root.AddPair('reader_licenses', Licenses);
    for Name in ReaderLicenseNames do
      Licenses.AddPair(Name, IncludeTrailingPathDelimiter(NativeParamStr(10)) + Name);
    for Name in ReaderProvenanceNames do
    begin
      LicenseFile := IncludeTrailingPathDelimiter(NativeParamStr(10)) + Name;
      if FileExists(LicenseFile) then Licenses.AddPair(Name, LicenseFile);
    end;
    Kit := TJSONObject.Create;
    Root.AddPair('kit', Kit);
    Kit.AddPair('compiler', NativeParamStr(11));
    Kit.AddPair('linker', NativeParamStr(12));
    Kit.AddPair('ffi_library', NativeParamStr(13));
    Kit.AddPair('msvc_libraries', NativeParamStr(14));
    Kit.AddPair('ucrt_libraries', NativeParamStr(15));
    Kit.AddPair('windows_libraries', NativeParamStr(16));
    Kit.AddPair('nvrtc_library', NativeParamStr(17));
    Result := Root;
    Root := nil;
  finally
    Root.Free;
    IdentityData.Free;
  end;
end;

procedure Run;
var
  Data: TJSONValue;
  Request, Inspection, VisionInfo: TJSONObject;
  Config: TQwen35Config;
  Logger: TLogger;
  Inspect, Launcher: Boolean;
begin
  Inspect := (NativeParamCount = 2) and (NativeParamStr(1) = '--inspect');
  Launcher := (NativeParamStr(1) = '--launcher') or (NativeParamStr(1) = '--launcher-inspect');
  Inspect := Inspect or (NativeParamStr(1) = '--launcher-inspect');
  RequireCpu((NativeParamCount = 1) or Inspect or Launcher, 'Usage: ConvertQwen35 [--inspect] request.json');
  if Launcher then
    Data := LauncherRequest
  else
    Data := ReadModelJSON(NativeParamStr(NativeParamCount));
  Logger := TLogger.Create;
  try
    Request := ModelObject(Data, 'Qwen conversion request');
    if Inspect then
    begin
      Inspection := InspectQwen35TextSource(Request, Config);
      try
        if Launcher then
        begin
          WriteLn('Dense Qwen 3.5/3.8 native TVM decoder');
          WriteLn(
            'Layers: ',
            Config.Layers,
            ' (',
            ModelInt(
              Inspection.GetValue('linear_attention_layers'),
              'linear'
            ),
            ' linear, ',
            ModelInt(
              Inspection.GetValue('full_attention_layers'),
              'full'
            ),
            ' full attention)'
          );
          WriteLn('Hidden: ', Config.Hidden, '; vocabulary: ', Config.Vocab);
          WriteLn('Compiled context: ', ModelInt(Inspection.GetValue('context'), 'context'));
          WriteLn('Source weight files: ', ModelObject(Inspection.GetValue('source_files'), 'sources').Count);
          WriteLn('Source convention: ', UTF8Encode(ModelText(Inspection.GetValue('source_convention'))));
          if JSONIsTrue(Inspection.GetValue('vision_supported')) then
          begin
            VisionInfo := ModelObject(Inspection.GetValue('vision_encoder'), 'vision encoder');
            WriteLn('Inputs: text and images');
            WriteLn(
              'Image canvas: ',
              ModelInt(
                VisionInfo.GetValue('image_width'),
                'image width'
              ),
              ' x ',
              ModelInt(
                VisionInfo.GetValue('image_height'),
                'image height'
              ),
              ' pixels (width x height)'
            );
            WriteLn('Original BF16 vision tensors: ',
              ModelInt(VisionInfo.GetValue('weight_tensors'), 'vision tensors'),
              ' (names and shapes match the vision graph; no image was encoded)');
          end
          else
            WriteLn('Inputs: text');
          WriteLn('Original source bytes are preserved; no weight requantization.');
          WriteLn('The text decoder uses Gated DeltaNet and gated full attention.');
          WriteLn('Layer streaming loads the source weights needed for each layer.');
        end
        else
          WriteLn(UTF8Encode(Inspection.ToJSON));
      finally
        Inspection.Free;
      end;
    end
    else
      PackageQwen35Text(Request, Logger.Progress);
  finally
    Logger.Free;
    Data.Free;
  end;
end;

begin
  SetTextCodePage(Output, 65001);
  SetTextCodePage(ErrOutput, 65001);
  try
    Run;
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.ClassName + ': ' + E.Message));
      ExitCode := 1;
    end;
  end;
end.
