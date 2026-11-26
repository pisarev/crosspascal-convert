{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35Bundle                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35Bundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Qwen35Source, CrossPascal.TVMCompiler, QwenCuda;

{ Completes the source plan with three reusable decoder graph families and,
  when the context holds a chunk, two optional batched prefill graphs.
  Packaging unchanged source files/tokenizer and their hashes is a separate
  step. The caller must use a fresh destination for each conversion attempt. }
procedure CompileQwen35TextGraphs(const Manifest: TJSONObject; const Config: TQwen35Config;
  const Destination: string; const Kit: TWindowsTvmKit; const NvrtcLibrary, Architecture: string;
  const Progress: TQwenProgress = nil);

implementation

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF}, CrossPascal.Qwen35Graph,
  CrossPascal.Qwen35PrefillGraph, CrossPascal.Onnx, CrossPascal.TVMOnnx, TVM.CpuGraph;

procedure CompileQwen35TextGraphs(const Manifest: TJSONObject; const Config: TQwen35Config;
  const Destination: string; const Kit: TWindowsTvmKit; const NvrtcLibrary, Architecture: string;
  const Progress: TQwenProgress);
const
  Names: array[0..2] of string = ('linear', 'full', 'head');
  PrefillChunk = 16;
var
  Graphs, Prefill: TJSONObject;
  Convention: TQwen35WeightConvention;
  Storage: TQwen35WeightStorage;
  Model: TOnnxModel;
  I, Context, Rows: Integer;
  Name, LibraryName: string;

  { Compiles Graph into FileName, records its interface as Owner.Key and
    frees Graph. }
  procedure CompileLayout(const Owner: TJSONObject; const Key, FileName: string;
    const Graph: TOnnxModel);
  var
    Layout, Entry: TJSONObject;
    Inputs, Outputs, Shape: TJSONArray;
    Input: TOnnxValue;
    Dimension: TOnnxDimension;
  begin
    try
      CompileWindowsTvmOnnxLibrary(Graph, Kit, BundleMember(Destination, FileName),
        NvrtcLibrary, Architecture);
      Layout := TJSONObject.Create;
      Owner.AddPair(Key, Layout);
      Layout.AddPair('library', FileName);
      Inputs := TJSONArray.Create;
      Outputs := TJSONArray.Create;
      Layout.AddPair('inputs', Inputs);
      Layout.AddPair('outputs', Outputs);
      for Input in Graph.Inputs do
      begin
        Entry := TJSONObject.Create;
        Inputs.AddElement(Entry);
        Entry.AddPair('name', Input.Name);
        Entry.AddPair('dtype', JSONInteger(Input.DataType));
        Shape := TJSONArray.Create;
        Entry.AddPair('shape', Shape);
        for Dimension in Input.Shape do Shape.AddElement(JSONInteger(Dimension.Value));
      end;
      for Input in Graph.Outputs do Outputs.AddElement(TJSONString.Create(Input.Name));
    finally
      Graph.Free;
    end;
  end;
begin
  RequireCpu((Manifest <> nil) and (Manifest.GetValue('graphs') = nil) and
    (Manifest.GetValue('prefill_graphs') = nil), 'Qwen graphs already exist');
  RequireCpu(DirectoryExists(Destination), 'Qwen graph destination does not exist');
  Name := ModelText(Manifest.GetValue('source_convention'));
  if Name = 'original' then
    Convention := qwOriginal
  else begin
    RequireCpu(Name = 'gguf', 'Unknown Qwen source convention');
    Convention := qwGGUF;
  end;
  Name := ModelText(Manifest.GetValue('weight_storage'));
  if Name = 'bfloat16' then
    Storage := qwsBF16
  else begin
    RequireCpu(Name = 'float32', 'Unknown Qwen graph storage');
    Storage := qwsFloat32;
  end;
  Name := ModelText(Manifest.GetValue('cuda_projection_schedule'));
  if Name <> '' then
  begin
    RequireCpu((Name = 'row_reduce_128') and (NvrtcLibrary <> '') and
      (Storage = qwsFloat32), 'Invalid CUDA FP32 projection schedule');
    Storage := qwsFloat32Row;
  end;
  Context := ModelInt(Manifest.GetValue('context'), 'compiled context');
  Rows := ModelInt(Manifest.GetValue('head_rows'), 'output rows');
  RequireCpu((Context > 0) and (Context <= Config.Context) and
    (Rows > 0) and (Rows <= Config.Vocab), 'Invalid Qwen graph dimensions');
  Graphs := TJSONObject.Create;
  Manifest.AddPair('graphs', Graphs);
  if NvrtcLibrary = '' then
    Manifest.AddPair('device', 'cpu')
  else begin
    RequireCpu(Architecture <> '', 'CUDA architecture is required');
    Manifest.AddPair('device', 'cuda');
    Manifest.AddPair('cuda_arch', Architecture);
  end;
  for I := 0 to High(Names) do
  begin
    Name := Names[I];
    LibraryName := Name + '.dll';
    RequireCpu(not FileExists(BundleMember(Destination, LibraryName)), 'Qwen graph output already exists');
    if Assigned(Progress) then Progress('Compiling Qwen ' + Name + ' graph...');
    case I of
      0: Model := BuildQwen35LayerGraph(Config, qaGatedDeltaNet, Context, Convention, Storage);
      1: Model := BuildQwen35LayerGraph(Config, qaFullAttention, Context, Convention, Storage);
    else
      Model := BuildQwen35HeadGraph(Config, Rows, Convention, Storage);
    end;
    CompileLayout(Graphs, Name, LibraryName, Model);
  end;
  { Batched prefill over PrefillChunk rows with the same parameters, states,
    convention, storage, device and architecture. The manifest field is added
    only after both libraries have compiled. }
  if Context < PrefillChunk then Exit;
  Prefill := TJSONObject.Create;
  try
    Prefill.AddPair('chunk_size', JSONInteger(PrefillChunk));
    for I := 0 to 1 do
    begin
      Name := Names[I];
      LibraryName := 'prefill-' + Name + '.dll';
      RequireCpu(not FileExists(BundleMember(Destination, LibraryName)), 'Qwen graph output already exists');
      if Assigned(Progress) then Progress('Compiling Qwen ' + Name + ' prefill graph...');
      if I = 0 then
        Model := BuildQwen35PrefillGraph(Config, qaGatedDeltaNet, Context, PrefillChunk, Convention, Storage)
      else
        Model := BuildQwen35PrefillGraph(Config, qaFullAttention, Context, PrefillChunk, Convention, Storage);
      CompileLayout(Prefill, Name, LibraryName, Model);
    end;
    Manifest.AddPair('prefill_graphs', Prefill);
  except
    Prefill.Free;
    raise;
  end;
end;

end.
