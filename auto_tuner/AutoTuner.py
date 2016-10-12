import os
import sys
import time
import codecs
import datetime
import glob
import shutil
import subprocess
import multiprocessing
import math
import re
import itertools
from functools import partial
import pickle
import atexit
import TestCase
from AnalysisDatabase import *
from DotFileDecorator import *
import Configuration
import ParseUtils
from Enums import *

####################################################################################################
pipeline_config = None
scene_config = None
log_file = None
log_level = -1

####################################################################################################
def print_auto_tuning_duration(start):
    duration = time.time() - start
    log(LogLevel.INFO, "auto tuning took {0} second(s)".format(duration))

####################################################################################################
def multiprocessing_initialization(counter_, length_, log_queue_, control_):
    global counter
    counter = counter_
    global length
    length = length_
    global log_queue
    log_queue = log_queue_
    global control
    control = control_

####################################################################################################
def log(log_level_, message):
    if log_level_ < log_level:
        return
    log_entry = "{0} [{1}] - {2}".format(datetime.datetime.now(), LogLevel.NAMES[log_level_], message)
    print(log_entry)
    sys.stdout.flush()
    if log_file is not None:
        log_file.write(log_entry + "\n")
        log_file.flush()

####################################################################################################
def enqueue_log(log_level_, message):
    log_queue.put((log_level_, message))

####################################################################################################
def log_queue_listener(log_queue, log_filename, log_level_):
    global log_level
    log_level = log_level_
    global log_file
    log_file = codecs.open(log_filename, "a+", "utf-8")
    with control.get_lock():
        control.value = 0
    while True:
        while not log_queue.empty():
            log_level, message = log_queue.get()
            log(log_level, message)
        with control.get_lock():
            end = control.value != 0 and log_queue.empty()
        if end:
            break

####################################################################################################
def open_process(args, bufsize=1, executable=None, stdin=None, stdout=None, stderr=None, 
                 preexec_fn=None, close_fds=False, shell=False, cwd=None, env=None, universal_newlines=False, 
                 startupinfo=None, creationflags=0, timeout = 0):
    proc = subprocess.Popen(args=args,
                            bufsize=bufsize,
                            executable=executable,
                            stdin=stdin,
                            stdout=stdout,
                            stderr=stderr,
                            preexec_fn=preexec_fn,
                            close_fds=close_fds,
                            shell=shell,
                            cwd=cwd,
                            env=env,
                            universal_newlines=universal_newlines,
                            startupinfo=startupinfo,
                            creationflags=creationflags)
    max_num_attempts = timeout * 10
    attempts = 0
    returncode = None
    while attempts < max_num_attempts:
        returncode = proc.poll()
        if returncode is not None:
            break
        time.sleep(0.1)
        attempts += 1
    if returncode is None:
        proc.kill()
    return returncode

####################################################################################################
def load_connections_map(partition_name):
    tokens = partition_name.split("_")
    partition_name = "_".join(tokens[0:3] + tokens[4:])
    assert os.path.exists(scene_config.outputPath + partition_name + ".conn")
    conns_map = {}
    with codecs.open(scene_config.outputPath + partition_name + ".conn", "r", "utf-8") as file:
        lines = file.readlines()
    for line in lines:
        if len(line) == 0:
            continue
        tokens1 = line.split("=")
        assert len(tokens1) == 2
        vertexIdx = int(tokens1[0])
        tokens2 = tokens1[1].split(",")
        assert len(tokens2) > 0
        conns_map[vertexIdx] = [int(t) for t in tokens2]
    return conns_map

####################################################################################################
def get_partition_idx_and_uid_from_result_name(result_name):
    tokens = result_name.split("_")
    partition_name = "_".join(tokens[0:2] + tokens[3:])
    pattern1 = re.compile("Idx\s=\s([0-9]+)")
    pattern2 = re.compile("Uid\s=\s\\\"([0-1]+)\\\"")
    source_code_filename = "{0}{1}.cuh".format(scene_config.outputPath, partition_name)
    with codecs.open(source_code_filename, "r", "utf-8") as file: 
        source_code = file.read()
        g1 = pattern1.findall(source_code)
        if len(g1) == 0:
            raise Exception("couldn't find idx in source code (pattern1=\"{0}\", source_code_filename=\"{1}\")".format(pattern1, source_code_filename))
        idx = g1[0]
        g2 = pattern2.findall(source_code)
        if len(g2) == 0:
            raise Exception("couldn't find uid in source code (pattern2=\"{0}\", source_code_filename=\"{0}\")".format(pattern2, source_code_filename))
        uid = g2[0]
        return (idx, uid)
    raise Exception("couldn't find partition idx or uid")

####################################################################################################
def call_analyzer(optimization, instrumentation, puid = None):   
    if not os.path.isfile(pipeline_config.analyzerBin):
        raise Exception("operator graph analyzer couldn't be found: {0}".format(pipeline_config.analyzerBin))
    args = [pipeline_config.analyzerBin, 
            "path=" + scene_config.grammarFile, 
            "templ=" + scene_config.templateFile, 
            "out=" + scene_config.outputPath,
            "" if puid == None else "puid=" + puid,
            "mvert=" + str(scene_config.maxNumVertices), 
            "mind=" + str(scene_config.maxNumIndices), 
            "minst=" + str(scene_config.maxNumElements), 
            "qmem=" + str(scene_config.queuesMem), 
            "opt=" + str(optimization), 
            "instr=" + ("1" if instrumentation else "0"), 
            "gridx=" + str(scene_config.gridX), 
            "gridy=" + str(scene_config.gridY), 
            "isize=" + str(scene_config.itemSize),
            "h1to1=" + str(scene_config.H1TO1).lower(),
            "hrr=" + str(scene_config.HRR).lower(),
            "hsto=" + str(scene_config.HSTO),
            "htspecs=" + scene_config.HTSPECS,
            "rsampl=" + str(scene_config.RSAMPL),
            "seed=" + str(scene_config.analyzerSeed),
            "tech=" + str(scene_config.whippletreeTechnique)]
    return_value = open_process(args=args, timeout=pipeline_config.analyzerExecutionTimeout)
    if return_value != 0:
        log(LogLevel.FATAL, "analyzer failed with return code {0} (args={1})".format(return_value, " ".join(args)))
        sys.exit(-1);

####################################################################################################
def renumber_partition_files(i, j, optimization, instrumentation):
    if i == j:
        return
    pattern = scene_config.outputPath + "partition_{0}" + ("_o" + str(optimization)) + ("_i" if instrumentation else "")
    base_old_partition = pattern.format(i)
    base_new_partition = pattern.format(j)
    old_conn_file = "{0}.conn".format(base_old_partition)
    new_conn_file = "{0}.conn".format(base_new_partition)
    old_cuh_file = "{0}.cuh".format(base_old_partition)
    new_cuh_file = "{0}.cuh".format(base_new_partition)
    old_dot_file = "{0}.dot".format(base_old_partition)
    new_dot_file = "{0}.dot".format(base_new_partition)
    if os.path.exists(old_conn_file):
        if os.path.exists(new_conn_file):
            os.remove(new_conn_file)
        os.rename(old_conn_file, new_conn_file)
    if os.path.exists(old_cuh_file):
        if os.path.exists(new_cuh_file):
            os.remove(new_cuh_file)
        with open(old_cuh_file, "r") as file:
            content = file.read()
        os.remove(old_cuh_file)
        content = content.replace("partition_{0}".format(i), "partition_{0}".format(j))
        content = content.replace("Idx = {0}".format(i), "Idx = {0}".format(j))
        with open(new_cuh_file, "w") as file:
            file.write(content)
    if os.path.exists(old_dot_file):
        if os.path.exists(new_dot_file):
            os.remove(new_dot_file)
        os.rename(old_dot_file, new_dot_file)

####################################################################################################
def generate_specific_partitions(puids):
    l = len(puids) - 1
    for i, puid in enumerate(puids):
        j = l - i
        call_analyzer(scene_config.optimization, scene_config.instrumentation, puid)
        renumber_partition_files(0, j, scene_config.optimization, scene_config.instrumentation)

####################################################################################################
def generate_all_partitions():
    call_analyzer(scene_config.optimization, scene_config.instrumentation)

####################################################################################################
def compile_partition(get_idx_func, num_partitions, partition, failed_comp, output_path, rebuild, compile_script, partition_bin, keep_temp, partition_compilation_timeout, log_func):
    workspace_path = "{0}{1}".format(output_path, partition)
    if os.path.exists("{0}/compilation_successful".format(workspace_path)) and not rebuild:
        log_func(LogLevel.WARNING, "[{0}/{1}] partition {2} already compiled".format(get_idx_func(), num_partitions, partition))
        return
    if (rebuild):
        build_param = "rebuild"
    else:
        build_param = "build"
    compilation_start = time.time()
    with open("{0}/compile_script.log".format(workspace_path), "w+") as logFile:
        return_value = open_process(args=[compile_script, 
                                   workspace_path, 
                                   build_param],
                                   stdout=logFile,
                                   stderr=logFile,
                                   timeout=partition_compilation_timeout)
    compilation_end = time.time()
    if return_value == 0:
        ori_partition_binary = "{0}{1}".format(workspace_path, partition_bin)
        partition_binary = "{0}bin/{1}.exe".format(output_path, partition)
        if not os.path.exists(ori_partition_binary):
            log_func(LogLevel.FATAL, "[{0}/{1}] partition {2} compiled successfully but binary couldn't be found (partition_binary={3})".format(
                get_idx_func(), num_partitions, partition, ori_partition_binary))
            sys.exit(-1)
        shutil.copyfile(ori_partition_binary, partition_binary)
        open(output_path + partition + "/compilation_successful", 'w') 
        log_func(LogLevel.INFO, "[{0}/{1}] partition {2} compiled successfully (compilation_time={3})".format(get_idx_func(), num_partitions, partition, compilation_end - compilation_start))
        # cleanup build dir to save disk space
        if not keep_temp:
            shutil.rmtree(output_path + partition + "\\build\\vs2013\\bin", True)
            shutil.rmtree(output_path + partition + "\\build\\vs2013\\build", True)
            os.remove(output_path + partition + "\\build\\vs2013\\vc120.pdb")
    else:             
        log_func(LogLevel.ERROR, "[{0}/{1}] partition {2} failed to compile (return_value={3}, compilation_time={4})".format(
            get_idx_func(), num_partitions, partition, return_value, compilation_end - compilation_start))
        if os.path.exists(output_path + partition + "/compilation_successful"):
            os.remove(output_path + partition + "/compilation_successful")
        failed_comp.append(partition)

####################################################################################################
def compile(partitions, failed_comp):
    num_partitions = len(partitions)
    for idx, partition in enumerate(partitions):
        def get_idx():
            return idx
        compile_partition(get_idx, 
                          num_partitions, 
                          partition, 
                          failed_comp, 
                          scene_config.outputPath, 
                          pipeline_config.rebuild, 
                          pipeline_config.compileScript, 
                          pipeline_config.partitionBin, 
                          pipeline_config.keepTemp, 
                          pipeline_config.partitionCompilationTimeout,
                          log)
    
####################################################################################################
def multiprocess_compile(partition, failed_comp, output_path, rebuild, compile_script, partition_bin, keep_temp, partition_compilation_timeout, log_func):
    def get_idx():
        with counter.get_lock():
            counter.value += 1
            idx = counter.value
        return idx
    compile_partition(get_idx, 
                      length, 
                      partition, 
                      failed_comp, 
                      output_path, 
                      rebuild, 
                      compile_script, 
                      partition_bin, 
                      keep_temp, 
                      partition_compilation_timeout, 
                      log_func)

####################################################################################################
def search_failed_executions(partition, failed_executions):
    for failed_execution in failed_executions:
        if partition == failed_execution[0]:
            return True
    return False

####################################################################################################
def process_results(partitions_executions, results, failed_results):
    num_executions = len(partitions_executions)
    for idx, (partition, execution) in enumerate(partitions_executions):
        cache_path = "{0}{1}{2}".format(scene_config.outputPath, pipeline_config.cachePath, partition)
        execution_object = TestCase.Execution.loadFromCache(cache_path, execution, scene_config.numRuns)
        if execution_object == None:
            failed_results.append(partition)
            continue
        log(LogLevel.INFO, "[{0}/{1}] execution {2} processed successfully (generation_time={3})".format(idx, num_executions, execution, execution_object.generationTime))
        if execution_object.metadata.instrumented:
            conn_map = load_connections_map(execution)
            execution_object.computeSubGraphExecTime(conn_map)
            execution_object.setEdgeSources(conn_map)
        results.append(execution_object)

####################################################################################################
def multiprocess_process_results(partition_execution, results, failed_results, output_path, num_runs, pipeline_cache_path, log_func):
    partition = partition_execution[0]
    execution = partition_execution[1]
    cache_path = "{0}{1}{2}".format(output_path, pipeline_cache_path, partition)
    try:
        execution_object = TestCase.Execution.loadFromCache(cache_path, execution, num_runs)
        with counter.get_lock():
            counter.value += 1
            idx = counter.value
        log_func(LogLevel.INFO, "[{0}/{1}] execution {2} processed successfully (generation_time={3})".format(idx, length, execution, execution_object.generationTime))
        if execution_object.metadata.instrumented:
            conn_map = load_connections_map(execution)
            execution_object.computeSubGraphExecTime(conn_map)
            execution_object.setEdgeSources(conn_map)
        results.append(execution_object)
    except TestCase.LoadingError as e:
        failed_results.append(partition)
        with counter.get_lock():
            counter.value += 1
            idx = counter.value
        log_func(LogLevel.ERROR, "[{0}/{1}] execution {2} processing failed [exception={3}]".format(idx, length, execution, str(e)))

####################################################################################################
def save_to_database(processed_results, lightweight_partitions):
    idx_to_unique_key = []
    unique_keys = []
    skip_idxs = []
    for result in processed_results:
        idx, uid = get_partition_idx_and_uid_from_result_name(result.name)
        partition_name = "partition_{0}_{1}{2}".format(idx, result.num_axioms, ("_o" + str(result.metadata.optimization)) + ("_i" if result.metadata.instrumented else ""))
        unique_key = [uid, result.num_axioms, result.metadata.optimization, result.metadata.instrumented]
        if unique_key in unique_keys:
            i = unique_keys.index(unique_key)
            log(LogLevel.WARNING, "unique key conflict between partitions {0} and {1}".format(partition_name, idx_to_unique_key[i]))
            skip_idxs.append(idx)
            continue
        idx_to_unique_key.append(partition_name)
        unique_keys.append(unique_key)

    database_name = "{0}analysis".format(scene_config.outputPath)
    Database.create(database_name, pipeline_config.databaseScript)
    Database.open(database_name)
    partition_counter = 0
    edge_counter = 0

    subgraph_counter = 0
    count = 0
    lightweight_partitions = []
    log(LogLevel.INFO, "saving {0} processed results to database".format(len(processed_results)))
    # entering transactional mode
    Database.connection.isolation_level = "DEFERRED"
    for result in processed_results:
        idx, uid = get_partition_idx_and_uid_from_result_name(result.name)
        if idx in skip_idxs:
            continue
        partition_name = "partition_{0}_{1}{2}".format(idx, result.num_axioms, ("_o" + str(result.metadata.optimization) + ("_i" if result.metadata.instrumented else "")))
        count = count + 1
        log(LogLevel.INFO, "saving {0}...".format(partition_name))
        partition = Partition()
        partition.idx = idx
        partition.uid = uid
        lightweight_partitions.append(partition)
        partition.instr = result.metadata.instrumented
        partition.opt = result.metadata.optimization
        partition.num_edges = len(uid)
        partition.exec_time = result.generationTime
        partition.num_axioms = result.num_axioms
        partition.stdev = result.genTimeStDev
        PartitionDAO.insert(partition)
        if result.metadata.instrumented:
            subgraph_pks = {}
            for s in result.subGraphData:
                subgraph = Subgraph()
                subgraph.partition_pk = partition.pk
                subgraph.idx = s.idx
                subgraph.freq = s.num_traversals
                subgraph.exec_time = s.avg_exec_time
                subgraph.stdev = s.avg_exec_time_std_dev
                SubgraphDAO.insert(subgraph)
                assert not subgraph.idx in subgraph_pks
                subgraph_pks[subgraph.idx] = subgraph.pk
                subgraph_counter = subgraph_counter + 1
            for e in result.edgeData:
                edge = Edge()
                edge.partition_pk = partition.pk
                edge.idx = e.idx
                edge.type = e.call_type
                edge.freq = e.num_traversals
                edge.exec_time = e.avg_exec_time
                edge.stdev = e.avg_exec_time_std_dev     
                assert e.sg_idx != None
                edge.subgraph_pk = subgraph_pks[e.sg_idx]
                EdgeDAO.insert(edge)
                edge_counter = edge_counter + 1
        partition_counter = partition_counter + 1

    Database.connection.commit()
    # returning to autocommit mode
    Database.connection.isolation_level = None

    # FIXME: checking invariants
    assert PartitionDAO.count() == partition_counter
    assert EdgeDAO.count() == edge_counter
    assert SubgraphDAO.count() == subgraph_counter

    Database.close()

####################################################################################################
STAGES = ["generatePartitions",
          "prepareWorkspaces",
          "compile",
          "findMaxNumAxioms",
          "run",
          "processResults",
          "saveToDatabase"]

####################################################################################################
def get_stage_idx(stage):
    for i, j in enumerate(STAGES):
        if stage == j:
            return i
    return -1

####################################################################################################
def get_enabled_stages(from_stage, to_stage):
    i = get_stage_idx(from_stage)
    j = get_stage_idx(to_stage)
    if i == -1:
        i = 0
    if j == -1:
        j = len(STAGES) - 1
    return [(k >= i and k <= j) for k in range(0, len(STAGES))]

####################################################################################################
def dot_file_decoration(min_maxnumax, lightweight_partitions = None):
    # FIXME: checking invariants
    assert os.path.exists(scene_config.outputPath + "analysis.db")
    Database.open(scene_config.outputPath + "analysis")
    if lightweight_partitions == None:
        lightweight_partitions = PartitionDAO.lightWeightList()
    count = 0
    log(LogLevel.INFO, "decorating {0} partitions".format(len(lightweight_partitions)))
    for lightweight_partition in lightweight_partitions:
        count = count + 1
        if (count % 100) == 0:
            log(LogLevel.INFO, "{0}".format(count))
        if lightweight_partition.num_axioms == 1 or lightweight_partition.num_axioms == min_maxnumax: 
            if not decorate_dot_file(lightweight_partition.uid):
                continue
            # FIXME: checking invariants
            assert os.path.exists(scene_config.outputPath + "d_partition_" + str(lightweight_partition.num_axioms) + "_" + str(lightweight_partition.idx) + "_" + lightweight_partition.uid + "_o2_i.dot")
            assert os.path.exists(scene_config.outputPath + "d_partition_" + str(lightweight_partition.num_axioms) + "_" + str(lightweight_partition.idx) + "_" + lightweight_partition.uid +  "_o3_i.dot")
    Database.close()

####################################################################################################
####################################################################################################
####################################################################################################
def main():
    global pipeline_config
    global scene_config
    global log_file
    global log_level

    pipeline_config_filename = ""
    from_stage = None
    to_stage = None
    stage = None
    scene_config_filename = ""
    partition_indices = []
    ignored_partition_indices = []
    start_partition = -1
    end_partition = -1
    puids = []
    recover_max_num_axioms = False
    force_max_num_axioms = False
    skip_compilations = -1
    skip_runs = -1
    for i in range(1, len(sys.argv)):
        kvp = sys.argv[i].split("=")
        if len(kvp) < 2:
            continue
        if kvp[0] == "pipeline_config":
            pipeline_config_filename = kvp[1]
        elif kvp[0] == "from_stage":
            from_stage = kvp[1]
        elif kvp[0] == "to_stage":
            to_stage = kvp[1]
        elif kvp[0] == "stage":
            stage = kvp[1]
        elif kvp[0] == "scene_config":
            scene_config_filename = kvp[1]
        elif kvp[0] == "partitions_indices":
            partition_indices = ParseUtils.try_to_parse_int_list(kvp[1])
        elif kvp[0] == "ignored_partitions":
            ignored_partition_indices = ParseUtils.try_to_parse_int_list(kvp[1])
        elif kvp[0] == "start_partition":
            start_partition = ParseUtils.try_to_parse_int(kvp[1], 0)
        elif kvp[0] == "end_partition":
            end_partition = ParseUtils.try_to_parse_int(kvp[1], -1)
        elif kvp[0] == "puids":
            puids = ParseUtils.try_to_parse_str_list(kvp[1])
        elif kvp[0] == "skip_compilations":
            skip_compilations = ParseUtils.try_to_parse_int(kvp[1], -1)
        elif kvp[0] == "recover_max_num_axioms":
            recover_max_num_axioms = ParseUtils.try_to_parse_bool(kvp[1])
        elif kvp[0] == "force_max_num_axioms":
            force_max_num_axioms = ParseUtils.try_to_parse_bool(kvp[1])
        elif kvp[0] == "skip_runs":
            skip_runs = ParseUtils.try_to_parse_int(kvp[1], -1)

    if not os.path.exists(pipeline_config_filename):
        log(LogLevel.FATAL, "pipeline configuration file doesn't exists (pipeline_config_filename={0})".format(pipeline_config_filename))
        sys.exit(-1)

    if not os.path.exists(scene_config_filename):
        log(LogLevel.FATAL, "scene configuration file doesn't exists (scene_config_filename={0})".format(scene_config_filename))
        sys.exit(-1)

    # setup global configurations
    pipeline_config = Configuration.Pipeline()
    scene_config = Configuration.Scene()

    pipeline_config.loadFromDisk(pipeline_config_filename)
    scene_config.loadFromDisk(scene_config_filename)

    if (start_partition > end_partition) and not (end_partition == -1):
        log(LogLevel.FATAL, "start_partition must be greater than end_partition")
        sys.exit(-1)

    if (scene_config.gridX <= 0 or scene_config.gridY <= 0) and scene_config.maxNumAxioms <= 0:
        log(LogLevel.FATAL, "either maxNumAxioms or gridX and gridY must be greater than 0")
        sys.exit(-1)

    ##################################################
    ##################################################

    if not os.path.exists(scene_config.outputPath):
        os.makedirs(scene_config.outputPath)

    log_filename = "{0}/auto_tuner.log".format(scene_config.outputPath)
    log_file = codecs.open(log_filename, "a+", "utf-8")

    # setup global log level
    log_level = pipeline_config.logLevel

    log(LogLevel.INFO, "Auto tuning...")

    ##################################################
    ##################################################

    if from_stage != None or to_stage != None:
        enabled_stages_map = get_enabled_stages(from_stage, to_stage)
    else:
        enabled_stages_map = get_enabled_stages(stage, stage)
    
    # NOTE: converting from relative to absolute path and/or adding trailing slash
    scripts_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    if not os.path.isabs(pipeline_config.pgaBasePath):
        pipeline_config.pgaBasePath = os.path.join(os.path.abspath(scripts_dir + pipeline_config.pgaBasePath), '')
    if not os.path.isabs(pipeline_config.analyzerBin):
        pipeline_config.analyzerBin = os.path.abspath(scripts_dir + pipeline_config.analyzerBin)
    if not os.path.isabs(pipeline_config.binTemplatePath):
        pipeline_config.binTemplatePath = os.path.join(os.path.abspath(scripts_dir + pipeline_config.binTemplatePath), '')
    if not os.path.isabs(pipeline_config.workspaceTemplatePath):
        pipeline_config.workspaceTemplatePath = os.path.join(os.path.abspath(scripts_dir + pipeline_config.workspaceTemplatePath), '')
    if not os.path.isabs(pipeline_config.databaseScript):
        pipeline_config.databaseScript = os.path.abspath(scripts_dir + pipeline_config.databaseScript)
    if not os.path.isabs(pipeline_config.compileScript):
        pipeline_config.compileScript = os.path.abspath(scripts_dir + pipeline_config.compileScript)
    if not os.path.isabs(pipeline_config.compileDependenciesScript):
        pipeline_config.compileDependenciesScript = os.path.abspath(scripts_dir + pipeline_config.compileDependenciesScript)
    scene_config.outputPath = os.path.join(scene_config.outputPath, '')
    
    if not os.path.exists("{0}tmp".format(scene_config.outputPath)):
        os.mkdir("{0}tmp".format(scene_config.outputPath))

    partitions = []
    max_num_axioms_per_partitions = []
    partitions_executions = []
    manager = multiprocessing.Manager()
    failed_compilations = manager.list()
    failed_executions = manager.list()
    failed_results = manager.list()
    processed_results = manager.list()

    max_num_axioms = scene_config.maxNumAxioms
    if max_num_axioms <= 0:
        max_num_axioms = scene_config.gridX * scene_config.gridY

    if scene_config.axiomStep < 0:
        axiom_step = int(math.ceil((max_num_axioms) / 4.0))
    else:
        axiom_step = scene_config.axiomStep
    
    if pipeline_config.automaticPoolSize:
        # NOTE: rule of thumb is num cores + 1 (dividing by two because of hyperthreading)
        pipeline_config.poolSize = int((multiprocessing.cpu_count() / 2) + 1)
        log(LogLevel.INFO, "using pool size of {0}".format(pipeline_config.poolSize))

    atexit.register(print_auto_tuning_duration, time.time())

    ##############################
    # [STAGE] GENERATE PARTITIONS
    ##############################
    if pipeline_config.generatePartitions and enabled_stages_map[0]:
        if len(puids) == 0:
            generate_all_partitions()
        else:
            generate_specific_partitions(puids)
        
    #######################
    # SELECTING PARTITIONS
    #######################
    if os.path.exists(scene_config.outputPath):
        paths = glob.glob(scene_config.outputPath + "*.cuh")
        if start_partition != -1:
            if end_partition != -1:
                partition_indices = range(start_partition, end_partition + 1)
            else:
                partition_indices = range(start_partition, len(paths))
        elif len(partition_indices) == 0:
            partition_indices = range(0, len(paths))

        for path in paths:
            partition = os.path.basename(path)
            p0 = partition.find("_") + 1
            p1 = partition.find("_", p0)
            if p1 == -1:
                p1 = partition.find(".", p0)
            partition_idx = int(partition[p0:p1])
            if (len(partition_indices) == 0 or partition_idx in partition_indices) and partition_idx not in ignored_partition_indices:
                if partition.find("_o{0}".format(scene_config.optimization)) != -1 and \
                    (not scene_config.instrumentation or partition.find("_i") != -1):
                    partitions.append(partition[partition.rfind("\\") + 1:-4])

    num_partitions = len(partitions)

    #############################
    # [STAGE] PREPARE WORKSPACES
    #############################
    if pipeline_config.prepareWorkspaces and enabled_stages_map[1]:
        source_path = "{0}src/test_runner/".format(pipeline_config.pgaBasePath)
        bin_path = "{0}bin/".format(scene_config.outputPath)
        if not os.path.exists(bin_path):
            shutil.copytree(pipeline_config.binTemplatePath, bin_path)
            if os.path.exists("bin/"):
                for filename in os.listdir("bin/"):
                    full_filename = "bin/{0}".format(filename)
                    if os.path.isfile(full_filename):
                        shutil.copyfile(full_filename, "{0}/{1}".format(bin_path, filename))
                    else:
                        shutil.copytree(full_filename, "{0}/{1}".format(bin_path, filename))
        for partition in partitions:
            workspace_path = "{0}{1}/".format(scene_config.outputPath, partition)
            if not os.path.exists(workspace_path):
                shutil.copytree(pipeline_config.workspaceTemplatePath, workspace_path)
                project_filename = "{0}build/vs2013/partition_test.vcxproj".format(pipeline_config.workspaceTemplatePath)
                with open(project_filename, "r") as project_file:
                    file_content = project_file.read()
                project_filename = "{0}build/vs2013/partition_test.vcxproj".format(workspace_path)
                with open(project_filename, "w") as project_file:
                    file_content = file_content.replace("#pgaBasePath#", pipeline_config.pgaBasePath)
                    file_content = file_content.replace("#cudaCC#", pipeline_config.cudaComputeCapability)
                    project_file.write(file_content)
                shutil.copyfile("{0}{1}.cuh".format(scene_config.outputPath, partition), "{0}Import.cuh".format(workspace_path))
            else:
                log(LogLevel.WARNING, "workspace directory for partition {0} already exists (workspace_path={1})".format(partition, workspace_path))

    ##################
    # [STAGE] COMPILE
    ##################
    if pipeline_config.compile and enabled_stages_map[2]:
        if pipeline_config.compileDependencies:
            dep_comp_start = time.time()
            return_value = open_process(args=[pipeline_config.compileDependenciesScript, 
                                             pipeline_config.pgaBasePath, 
                                             scene_config.outputPath], 
                                        timeout=pipeline_config.dependenciesCompilationTimeout)
            dep_comp_end = time.time()
            if return_value != 0:
                log(LogLevel.FATAL, "failed to compile dependencies (return_value={0}, dep_comp_time={1})".format(return_value, dep_comp_end - dep_comp_start))
                sys.exit(-1)

        if skip_compilations > 0:
            compilations = partitions[skip_compilations:]
            log(LogLevel.WARNING, "skipping {0} partition compilations".format(skip_compilations))
        else:
            compilations = list(partitions)
        num_compilations = len(compilations)
        log(LogLevel.INFO, "compiling {0} partition(s)".format(num_compilations))
        if pipeline_config.useMultiprocess == True:
            counter = multiprocessing.Value('i', 0)
            log_queue = manager.Queue()
            control = multiprocessing.Value('i', 0)
            pool = multiprocessing.Pool(processes=pipeline_config.poolSize,
                                        initializer=multiprocessing_initialization,
                                        initargs=(counter, num_compilations, log_queue, control,))
            pool.apply_async(log_queue_listener, (log_queue, log_filename, log_level,))
            map_func = partial(multiprocess_compile, 
                               failed_comp=failed_compilations, 
                               output_path=scene_config.outputPath, 
                               rebuild=pipeline_config.rebuild, 
                               compile_script=pipeline_config.compileScript, 
                               partition_bin=pipeline_config.partitionBin, 
                               keep_temp=pipeline_config.keepTemp,
                               partition_compilation_timeout=pipeline_config.partitionCompilationTimeout,
                               log_func=enqueue_log)
            pool.map(map_func, compilations)
            with control.get_lock():
                control.value = 1
            pool.close()
        else:
            compile(compilations, failed_compilations)
    else:
        #################################
        # RECOVERING FAILED COMPILATIONS
        #################################
        for partition in partitions:
            workspace_path = "{0}{1}".format(scene_config.outputPath, partition)
            partition_binary = "{0}bin/{1}.exe".format(scene_config.outputPath, partition)
            if not os.path.exists("{0}/compilation_successful".format(workspace_path)) and not os.path.exists(partition_binary):
                failed_compilations.append(partition)
        log(LogLevel.WARNING, "{0} failed compilations recovered".format(len(failed_compilations)))
    
    ##########################################
    # [STAGE] FIND MIN./MAX. NUMBER OF AXIOMS
    ##########################################
    if force_max_num_axioms:
        log(LogLevel.WARNING, "forcing global maximum number of axioms to {0}, skipping search".format(scene_config.maxNumAxioms))
        global_max_num_axioms = scene_config.maxNumAxioms
    else:
        DEVNULL = open(os.devnull, 'w')
        find_max_num_axioms_filename = "{0}tmp/find_max_num_axioms".format(scene_config.outputPath)
        if pipeline_config.findMaxNumAxioms and enabled_stages_map[3] and not recover_max_num_axioms:
            partitions_to_search_max_num_axioms = []
            for partition in partitions:
                if partition in failed_compilations:
                    continue
                if partition.find("_o{0}".format(scene_config.optimization)) != -1 and \
                    (not scene_config.instrumentation or partition.find("_i") != -1):
                    partitions_to_search_max_num_axioms.append(partition)

            num_partitions = len(partitions_to_search_max_num_axioms)
            log(LogLevel.INFO, "searching maximum number of axioms for {0} partition(s)".format(num_partitions))
            global_max_num_axioms = max_num_axioms
            for idx, partition in enumerate(partitions_to_search_max_num_axioms):
                if global_max_num_axioms == 1:
                    log(LogLevel.WARNING, "global maximum number of axioms is 1, skipping search")
                    break
                return_value = open_process(args=[
                                            scene_config.outputPath + "bin/" + partition + ".exe", 
                                            str(pipeline_config.deviceId), 
                                            str(scene_config.runSeed),
                                            "auto",
                                            "true", 
                                            str(scene_config.numRuns),  
                                            str(scene_config.minNumAxioms), 
                                            str(max_num_axioms), 
                                            str(axiom_step)
                                            ], 
                                            stdout=DEVNULL,
                                            stderr=DEVNULL,
                                            cwd=scene_config.outputPath + "bin",
                                            timeout=pipeline_config.partitionExecutionTimeout)
                run_log_filename = "{0}bin/run.log".format(scene_config.outputPath)
                if return_value == ReturnValue.SUCCESS:
                    with codecs.open(run_log_filename, "r", "utf-8") as file: 
                        last_num_axioms = int(file.readline())
                    if scene_config.globalMinNumAxioms != -1 and scene_config.globalMinNumAxioms > last_num_axioms:
                        log(LogLevel.WARNING, "[{0}/{1}] ignoring partition {2} [last_num_axioms={3}, global_min_num_axioms={4}]".format(
                            idx, num_partitions, partition, last_num_axioms, scene_config.globalMinNumAxioms))
                    else:
                        max_num_axioms_per_partitions.append([partition, last_num_axioms])
                        if last_num_axioms < global_max_num_axioms:
                            global_max_num_axioms = last_num_axioms
                        log(LogLevel.INFO, "[{0}/{1}] maximum number of axioms for {2} found [last_num_axioms={3}]".format(
                            idx, num_partitions, partition, last_num_axioms))
                else:
                    return_value_str = ReturnValue.NAMES[return_value] if return_value in ReturnValue.NAMES else str(return_value)
                    with codecs.open(run_log_filename, "r", "utf-8") as file: 
                        last_num_axioms = int(file.readline())
                    last_num_axioms = max(1, last_num_axioms - axiom_step)
                    if last_num_axioms == 1 or (scene_config.globalMinNumAxioms != -1 and scene_config.globalMinNumAxioms > last_num_axioms):
                        log(LogLevel.ERROR, "[{0}/{1}] ignoring partition {2} [last_num_axioms={3}, global_min_num_axioms={4}, return_value={5}]".format(
                            idx, num_partitions, partition, last_num_axioms, scene_config.globalMinNumAxioms, return_value_str))
                        failed_executions.append(partition)
                    else:
                        max_num_axioms_per_partitions.append([partition, last_num_axioms])
                        if last_num_axioms < global_max_num_axioms:
                            global_max_num_axioms = last_num_axioms
                        log(LogLevel.WARNINNG, "[{0}/{1}] maximum number of axioms for {2} found, but application crashed [last_num_axioms={3}, return_value={4}]".format(
                            idx, num_partitions, partition, last_num_axioms, return_value_str))
            if os.path.exists(find_max_num_axioms_filename):
                os.remove(find_max_num_axioms_filename)
            with open(find_max_num_axioms_filename, "wb") as file:
                pickle.dump(max_num_axioms_per_partitions, file)
            log(LogLevel.INFO, "maxmimum number of axioms found, global maximum number of axioms set to {0}".format(global_max_num_axioms))
        else:
            ##############################
            # RECOVERING MAX. NUM. AXIOMS
            ##############################
            if os.path.exists(find_max_num_axioms_filename):
                with open(find_max_num_axioms_filename, "rb") as file:
                    max_num_axioms_per_partitions = pickle.load(file)
            else:
                log(LogLevel.FATAL, "find max. num. axioms file not found (find_max_num_axioms_filename={0})".format(find_max_num_axioms_filename))
                sys.exit(-1)

            if len(max_num_axioms_per_partitions) > 0:
                global_max_num_axioms = min(max_num_axioms_per_partitions, key=lambda x: x[1])[1]

            log(LogLevel.WARNING, "{0} maximum number of axioms recovered, global maximum number of axioms set to {1}".format(len(max_num_axioms_per_partitions), global_max_num_axioms))

    for partition in partitions:
        if partition not in failed_compilations:
            tokens = partition.split("_")
            if (scene_config.minNumAxioms == 0):
                execution_min = "_".join(tokens[0:2] + ["1"] + tokens[2:])
            else:
                execution_min = "_".join(tokens[0:2] + [str(scene_config.minNumAxioms)] + tokens[2:])
            partitions_executions.append((partition, execution_min))
            if scene_config.minNumAxioms != global_max_num_axioms and (scene_config.minNumAxioms == 0 and global_max_num_axioms > 1):
                for num_axioms in range(axiom_step, global_max_num_axioms + 1, axiom_step):
                    execution = "_".join(tokens[0:2] + [str(num_axioms)] + tokens[2:])
                    partitions_executions.append((partition, execution))

    ##############
    # [STAGE] RUN
    ##############
    failed_executions_filename = "{0}tmp/failed_executions".format(scene_config.outputPath)
    if pipeline_config.run and enabled_stages_map[4]:
        DEVNULL = open(os.devnull, 'w')
        runs = [partition for partition in partitions if partition not in failed_compilations]
        for partition in failed_compilations:
            failed_executions.append([partition, "failed compilation"])
        if skip_runs > 0:
            runs = runs[skip_runs:]
            log(LogLevel.WARNING, "skipping {0} partition runs".format(skip_runs))
        num_runs = len(runs)
        log(LogLevel.INFO, "running {0} partition(s)".format(num_runs))
        for idx, partition in enumerate(runs):
            partition_binary = "{0}bin/{1}.exe".format(scene_config.outputPath, partition)
            if os.path.isfile(partition_binary):
                message = "[{0}/{1}] running {2}...".format(idx, num_runs, partition)
                return_value = open_process(args=[
                                            partition_binary,
                                            str(pipeline_config.deviceId), 
                                            str(scene_config.runSeed), 
                                            "auto", 
                                            "false", 
                                            str(scene_config.numRuns), 
                                            str(scene_config.minNumAxioms), 
                                            str(global_max_num_axioms),
                                            str(axiom_step)],
                                            stdout=DEVNULL, 
                                            stderr=DEVNULL,
                                            cwd=scene_config.outputPath + "/bin",
                                            timeout=pipeline_config.partitionExecutionTimeout)
                if return_value == ReturnValue.SUCCESS:
                    tokens = partition.split("_")
                    partition_dir = "_".join(tokens)
                    base_partition_filename = "_".join(tokens[0:2] + [str(global_max_num_axioms)] + tokens[2:])
                    generation_filename = "{0}{1}{2}/{3}-generation.txt".format(scene_config.outputPath, pipeline_config.cachePath, partition_dir, base_partition_filename)
                    try:
                        os.stat(generation_filename)
                        log(LogLevel.INFO, "{0}done".format(message))
                    except Exception as ex:
                        log(LogLevel.ERROR, "{0}failed! (ex={1})".format(message, ex))
                        failed_executions.append([partition, "couldn't find generation file (generation_filename={0})".format(generation_filename)])
                else:
                    log(LogLevel.ERROR, "{0}failed! (return_value={1})".format(message, ReturnValue.NAMES[return_value] if return_value in ReturnValue.NAMES else str(return_value)))
            else:
                failed_executions.append([partition, "[{0}/{1}] couldn't find partition binary (partition_binary={2})".format(idx, num_runs, partition_binary)])

        with open(failed_executions_filename, "wb") as file:
            pickle.dump([failed_execution for failed_execution in failed_executions], file)
    else:
        ###############################
        # RECOVERING FAILED EXECUTIONS
        ###############################
        if os.path.exists(failed_executions_filename):
            with open(failed_executions_filename, "rb") as file:
                failed_executions_tmp = pickle.load(file)
            for failed_execution in failed_executions_tmp:
                failed_executions.append(failed_execution)
        else:
            log(LogLevel.FATAL, "failed executions file not found (failed_executions_filename={0})".format(failed_executions_filename))
            sys.exit(-1)

    ##########################
    # [STAGE] PROCESS RESULTS
    ##########################
    failed_results_filename = "{0}tmp/failed_results".format(scene_config.outputPath)
    processed_results_filename = "{0}tmp/processed_results".format(scene_config.outputPath)
    if pipeline_config.processResults and enabled_stages_map[5]:
        valid_partitions_executions = [(partition, execution) for (partition, execution) in partitions_executions 
                                           if partition not in failed_compilations and not search_failed_executions(partition, failed_executions)]
        num_execution_results = len(valid_partitions_executions)
        log(LogLevel.INFO, "processing {0} execution results".format(num_execution_results))
        if pipeline_config.useMultiprocess == True:            
            counter = multiprocessing.Value('i', 0)
            log_queue = manager.Queue()
            control = multiprocessing.Value('i', 0)
            pool = multiprocessing.Pool(processes=pipeline_config.poolSize,
                                        initializer=multiprocessing_initialization,
                                        initargs=(counter, num_execution_results, log_queue, control,))
            pool.apply_async(log_queue_listener, (log_queue, log_filename, log_level,))
            map_func = partial(multiprocess_process_results, 
                               results=processed_results, 
                               failed_results=failed_results, 
                               output_path=scene_config.outputPath, 
                               num_runs=scene_config.numRuns, 
                               pipeline_cache_path=pipeline_config.cachePath, 
                               log_func=enqueue_log)
            pool.map(map_func, valid_partitions_executions)
            with control.get_lock():
                control.value = 1
            pool.close()
        else:
            process_results(valid_partitions_executions, processed_results, failed_results)
        if os.path.exists(processed_results_filename):
            os.remove(processed_results_filename)
        with open(processed_results_filename, "wb") as file:
            pickle.dump([processed_result for processed_result in processed_results], file)
        if os.path.exists(failed_results_filename):
            os.remove(failed_results_filename)
        with open(failed_results_filename, "wb") as file:
            pickle.dump([failed_result for failed_result in failed_results], file)
    else:
        ###############################
        # RECOVERING PROCESSED RESULTS
        ###############################
        if os.path.exists(processed_results_filename):
            with open(processed_results_filename, "rb") as file:
                processed_results_tmp = pickle.load(file)
            for processed_result in processed_results_tmp:
                processed_results.append(processed_result)
        else:
            log(LogLevel.FATAL, "processed results file not found (processed_results_filename={0})".format(processed_results_filename))
            sys.exit(-1)
        if os.path.exists(failed_results_filename):
            with open(failed_results_filename, "rb") as file:
                failed_results_tmp = pickle.load(file)
            for failed_result in failed_results_tmp:
                failed_results.append(failed_result)
        else:
            log(LogLevel.FATAL, "failed results file not found (failed_results_filename={0})".format(failed_results_filename))
            sys.exit(-1)

    ###########################
    # [STAGE] SAVE TO DATABASE
    ###########################
    lightweight_partitions = None
    if pipeline_config.saveToDatabase and enabled_stages_map[6]:
        save_to_database(processed_results, lightweight_partitions)

####################################################################################################
if __name__ == "__main__":
    main()