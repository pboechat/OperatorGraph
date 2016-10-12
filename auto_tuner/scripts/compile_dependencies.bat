@echo off
setlocal

set LOG_FILE=%cd%\%2\compile_dependencies.log

cd %1

call "%VS120COMNTOOLS%"\vsvars32.bat

echo Compiling whippletree dependency...
MSBuild /m /t:rebuild /p:Platform=x64 /p:Configuration=Release ../build/deps/whippletree/whippletree.vcxproj > %LOG_FILE% 2>&1

echo Compiling libpng dependency...
MSBuild /m /t:rebuild /p:Platform=x64 /p:Configuration=Release ../build/deps/libpng/libpng.vcxproj > %LOG_FILE% 2>&1

echo Compiling zlib dependency...
MSBuild /m /t:rebuild /p:Platform=x64 /p:Configuration=Release ../build/deps/zlib/zlib.vcxproj > %LOG_FILE% 2>&1

echo Compiling pga rendering module...
MSBuild /m /t:rebuild /p:Platform=x64 /p:Configuration=Release ../build/src/rendering/pga_rendering.vcxproj > %LOG_FILE% 2>&1

exit /b %errorlevel%

endlocal