@echo off
setlocal
python ../../AutoTuner.py pipeline_config=../../pipeline.xml scene_config=scene_OPTsm.xml %*
endlocal