{ ************************************************************************** }
{                                                                            }
{ ConvertVisionOnnx                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

program ConvertVisionOnnx;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.VisionBundle;

var
  Kit: TWindowsTvmKit;
  Recipe: string;

begin
  if not (ParamCount in [10, 11]) then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertVisionOnnx checkpoint models.json profile-id destination');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib [graph-recipe.onnx]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := ParamStr(5);
    Kit.Linker := ParamStr(6);
    Kit.FfiLibrary := ExpandFileName(ParamStr(7));
    Kit.MsvcLibraries := ParamStr(8);
    Kit.UcrtLibraries := ParamStr(9);
    Kit.WindowsLibraries := ParamStr(10);
    Recipe := '';
    if ParamCount = 11 then Recipe := ParamStr(11);
    ConvertCpuVisionBundle(ParamStr(1), ParamStr(2), ParamStr(3), ParamStr(4), Kit, Recipe);
    WriteLn(UTF8Encode(ExpandFileName(ParamStr(4))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
