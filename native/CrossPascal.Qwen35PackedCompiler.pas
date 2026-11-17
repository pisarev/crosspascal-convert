{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35PackedCompiler                                            }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35PackedCompiler;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

procedure CompileQwen35PackedDecoder(const SourceDirectory, NvrtcLibrary, Architecture, OutputFile: string);

implementation

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  CrossPascal.Nvrtc, CrossPascal.SHA256, TVM.CpuGraph;

function ReadSource(const FileName, ExpectedSHA256: string): UTF8String;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= 1024 * 1024), 'Invalid tensor decoder source size');
    RequireCpu(SameText(SHA256Stream(Stream), ExpectedSHA256), 'Tensor decoder source changed');
    Stream.Position := 0;
    SetLength(Result, Stream.Size);
    Stream.ReadBuffer(Result[1], Length(Result));
  finally
    Stream.Free;
  end;
end;

procedure CompileQwen35PackedDecoder(const SourceDirectory, NvrtcLibrary, Architecture, OutputFile: string);
const
  Integers: UTF8String = '#pragma once'#10 +
    'typedef signed char int8_t; typedef unsigned char uint8_t;'#10 +
    'typedef short int16_t; typedef unsigned short uint16_t;'#10 +
    'typedef int int32_t; typedef unsigned int uint32_t;'#10 +
    'typedef long long int64_t; typedef unsigned long long uint64_t;'#10 +
    'namespace std { using ::int8_t; using ::uint8_t; using ::int16_t; using ::uint16_t;'#10 +
    'using ::int32_t; using ::uint32_t; using ::int64_t; using ::uint64_t; }'#10;
var
  Source, Header: UTF8String;
  Bytes: TBytes;
  Compiler: TNvrtcCompiler;
  Stream: TFileStream;
begin
  RequireCpu(not FileExists(OutputFile), 'Packed decoder output already exists');
  Source := ReadSource(BundleMember(SourceDirectory, 'packed_decode.cu'),
    '86c56d36a712b212fdd4e1006b7261689981e7c80efe18832397c40f9676cc8c');
  Header := ReadSource(BundleMember(SourceDirectory, 'ggml-common.h'),
    '0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c');
  Compiler := TNvrtcCompiler.Create(NvrtcLibrary, Architecture);
  try
    Bytes := Compiler.CompileRawKernel(Source, ['ggml-common.h', 'cstdint'], [Header, Integers]);
  finally
    Compiler.Free;
  end;
  Stream := TFileStream.Create(OutputFile, fmCreate);
  try
    Stream.WriteBuffer(Bytes[0], Length(Bytes));
  finally
    Stream.Free;
  end;
end;

end.
