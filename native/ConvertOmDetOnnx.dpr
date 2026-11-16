{ ************************************************************************** }
{                                                                            }
{ ConvertOmDetOnnx                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertOmDetOnnx;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.OmDetBundle;

var
  Kit: TWindowsTvmKit;
  Nvrtc, Architecture: string;
  RequiredEnd: Integer;

begin
  RequiredEnd := ParamCount;
  if (ParamCount = 11) and (ParamStr(9) = '--cuda') then
  begin
    RequiredEnd := 8;
    Nvrtc := ParamStr(10);
    Architecture := ParamStr(11);
  end;
  if RequiredEnd <> 8 then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertOmDetOnnx source-directory destination');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
    WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_75|sm_86|sm_89|sm_120]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := ParamStr(3);
    Kit.Linker := ParamStr(4);
    Kit.FfiLibrary := ExpandFileName(ParamStr(5));
    Kit.MsvcLibraries := ParamStr(6);
    Kit.UcrtLibraries := ParamStr(7);
    Kit.WindowsLibraries := ParamStr(8);
    ConvertOmDetBundle(ParamStr(1), ParamStr(2), Kit, Nvrtc, Architecture);
    WriteLn(UTF8Encode(ExpandFileName(ParamStr(2))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
