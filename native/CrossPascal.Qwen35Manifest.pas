{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35Manifest                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35Manifest;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Qwen35Source;

{ Describes the original, unchanged source payload. The bundle packager must
  materialize the listed flat files and add compiled graphs, tokenizer and
  artifact hashes before this plan can be opened by the runtime. }
function BuildQwen35TextManifest(const SourcePath, GGMLLibrary: string;
  const Context, HeadRows: Integer; out Config: TQwen35Config;
  const WithImages: Boolean = False): TJSONObject;

implementation

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  CrossPascal.SafeTensors, CrossPascal.GGUF, CrossPascal.Qwen35GGUF, TVM.CpuGraph;

function BuildQwen35TextManifest(const SourcePath, GGMLLibrary: string;
  const Context, HeadRows: Integer; out Config: TQwen35Config;
  const WithImages: Boolean): TJSONObject;
const
  Prefix = 'model.language_model.';
var
  Safe: TSafeTensorCheckpoint;
  GGUF: TGGUFCheckpoint;
  Data: TJSONValue;
  Root, ConfigJSON, Weights, Layer, LayerWeights, Descriptor, Files, FileInfo: TJSONObject;
  Layers, Shape: TJSONArray;
  BF16Bindings: TArray<TQwen35Binding>;
  GGUFBindings: TArray<TQwen35GGUFBinding>;
  BF16Binding: TQwen35Binding;
  GGUFBinding: TQwen35GGUFBinding;
  Storage: TSafeTensorStorage;
  Stream: TFileStream;
  Total, FileBytes: Int64;
  I, Index, Dot: Integer;
  Name, FileName: string;

  procedure BeginPlan(const Convention, Encoding: string);
  var
    I: Integer;
  begin
    RequireCpu((Context > 0) and (Context <= Config.Context), 'Invalid compiled Qwen context');
    RequireCpu((HeadRows > 0) and (HeadRows <= Config.Vocab), 'Invalid Qwen output tile size');
    Root.AddPair('schema', 'crosspascal.qwen35.text.v1');
    Root.AddPair('architecture', 'qwen3_5');
    Root.AddPair('source_convention', Convention);
    Root.AddPair('weight_storage', Encoding);
    Root.AddPair('context', JSONInteger(Context));
    Root.AddPair('head_rows', JSONInteger(HeadRows));
    Root.AddPair('text_only', JSONBoolean(not WithImages));
    ConfigJSON := TJSONObject.Create;
    Root.AddPair('config', ConfigJSON);
    ConfigJSON.AddPair('hidden', JSONInteger(Config.Hidden));
    ConfigJSON.AddPair('vocab', JSONInteger(Config.Vocab));
    ConfigJSON.AddPair('intermediate', JSONInteger(Config.Intermediate));
    ConfigJSON.AddPair('layers', JSONInteger(Config.Layers));
    Files := TJSONObject.Create;
    Root.AddPair('source_files', Files);
    Weights := TJSONObject.Create;
    Root.AddPair('weights', Weights);
    Layers := TJSONArray.Create;
    Root.AddPair('layers', Layers);
    for I := 0 to Config.Layers - 1 do
    begin
      Layer := TJSONObject.Create;
      Layers.AddElement(Layer);
      if Config.Attention[I] = qaGatedDeltaNet then
        Layer.AddPair('kind', 'linear')
      else
        Layer.AddPair('kind', 'full');
      Layer.AddPair('weights', TJSONObject.Create);
    end;
  end;

  procedure AddDescriptor(const LayerIndex: Integer; const Name, SourceName, FileName, Encoding: string;
    const Offset, Bytes, FileBytes: Int64; const Dimensions: TArray<Int64>; const Format: Integer = -1);
  var
    Dimension: Int64;
    Target: TJSONObject;
  begin
    BundleMember('', FileName);
    if Files.GetValue(FileName) = nil then
    begin
      FileInfo := TJSONObject.Create;
      FileInfo.AddPair('bytes', JSONInteger(FileBytes));
      Files.AddPair(FileName, FileInfo);
    end
    else
      RequireCpu(ModelInt(ModelObject(Files.GetValue(FileName), 'source file').GetValue('bytes'),
        'source file size') = FileBytes, 'Qwen source basenames collide');
    RequireCpu((LayerIndex >= -1) and (LayerIndex < Config.Layers), 'Invalid source layer index');
    if LayerIndex < 0 then
      Target := Weights
    else
      Target := ModelObject(ModelObject(Layers.Items[LayerIndex], 'layer').GetValue('weights'),
      'layer parameters');
    RequireCpu(Target.GetValue(Name) = nil, 'Duplicate Qwen parameter: ' + Name);
    Descriptor := TJSONObject.Create;
    Target.AddPair(Name, Descriptor);
    Descriptor.AddPair('source_name', SourceName);
    Descriptor.AddPair('file', FileName);
    Descriptor.AddPair('encoding', Encoding);
    Descriptor.AddPair('offset', JSONInteger(Offset));
    Descriptor.AddPair('bytes', JSONInteger(Bytes));
    Shape := TJSONArray.Create;
    Descriptor.AddPair('shape', Shape);
    for Dimension in Dimensions do Shape.AddElement(JSONInteger(Dimension));
    if Format >= 0 then Descriptor.AddPair('format', JSONInteger(Format));
  end;

begin
  Root := TJSONObject.Create;
  try
    if DirectoryExists(SourcePath) then
    begin
      Data := ReadModelJSON(BundleMember(SourcePath, 'config.json'));
      try
        Config := ReadQwen35Config(ModelObject(Data, 'Qwen configuration'));
      finally
        Data.Free;
      end;
      Safe := TSafeTensorCheckpoint.Create(SourcePath);
      try
        BF16Bindings := ReadQwen35BF16TextBindings(Safe, Config, Total);
        BeginPlan('original', 'bfloat16');
        Root.AddPair('text_payload_bytes', JSONInteger(Total));
        for BF16Binding in BF16Bindings do
        begin
          Name := BF16Binding.SourceName;
          Index := -1;
          if Copy(Name, 1, Length(Prefix)) = Prefix then Delete(Name, 1, Length(Prefix));
          if Copy(Name, 1, 7) = 'layers.' then
          begin
            Delete(Name, 1, 7);
            Dot := Pos('.', Name);
            RequireCpu((Dot > 1) and TryStrToInt(Copy(Name, 1, Dot - 1), Index),
              'Invalid Qwen source layer name');
            Delete(Name, 1, Dot);
          end;
          Storage := Safe.TensorStorage(BF16Binding.SourceName);
          AddDescriptor(Index, Name, BF16Binding.SourceName, ExtractFileName(Storage.FileName),
            'bf16', Storage.ByteOffset, Storage.ByteCount, Storage.FileSize, BF16Binding.Shape);
        end;
      finally
        Safe.Free;
      end;
    end
    else begin
      GGUF := TGGUFCheckpoint.Create(SourcePath, GGMLLibrary);
      try
        Config := ReadQwen35GGUFConfig(GGUF);
        GGUFBindings := ReadQwen35GGUFBindings(GGUF, Config);
        BeginPlan('gguf', 'float32');
        FileName := ExtractFileName(SourcePath);
        Stream := TFileStream.Create(SourcePath, fmOpenRead or fmShareDenyWrite);
        try
          FileBytes := Stream.Size;
        finally
          Stream.Free;
        end;
        for GGUFBinding in GGUFBindings do
          AddDescriptor(GGUFBinding.Layer, GGUFBinding.Name, GGUFBinding.SourceName, FileName,
            'gguf', GGUFBinding.Tensor.DataOffset, GGUFBinding.Tensor.DataBytes,
            FileBytes, GGUFBinding.Shape, GGUFBinding.Tensor.Format);
        Root.AddPair('dequantization', 'Original GGUF bytes; GGML tensor decode only; native TVM model math');
      finally
        GGUF.Free;
      end;
    end;
    for I := 0 to Layers.Count - 1 do
    begin
      LayerWeights := ModelObject(ModelObject(Layers.Items[I], 'layer').GetValue('weights'), 'weights');
      RequireCpu(LayerWeights.Count >= 11, 'Incomplete Qwen layer parameters');
    end;
    Result := Root;
    Root := nil;
  finally
    Root.Free;
  end;
end;

end.
