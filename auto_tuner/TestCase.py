import os
import sys
import codecs
import csv
import math
import re

def standard_deviation(values):
    M = 0.0
    S = 0.0
    k = 1
    for value in values:
        tmpM = M
        M += (value - tmpM) / k
        S += (value - tmpM) * (value - M)
        k += 1
    return math.sqrt(S / max(1, k - 2))

class Records:
    @staticmethod
    def arithmeticMean(values):
        sum = 0
        for v in values:
            sum += v
        return sum / len(values)

    @staticmethod
    def loadFromFileAsFloats(fileName):
        if not os.path.isfile(fileName):
            raise Exception("file not found {0}".format(fileName))
        file = open(fileName)
        records = [float(line) for line in file.readlines()]
        file.close()
        return records

class CuptiData:
    def __init__(self, fileName):
        if not os.path.isfile(fileName):
            raise Exception("file not found {0}".format(fileName))
        file = open(fileName)
        atom_counts = []
        gld = []
        gst = []
        for line in file.readlines():
            s = line.split()
            if len(s) > 0:
                if s[0] == "atom_count:":
                    atom_counts.append(int(float(s[1])))
                if s[0] == "gld_request:":
                    gld.append(int(float(s[1])))
                if s[0] == "gst_request:":
                    gst.append(int(float(s[1])))
        file.close()
        self.atom_count = Records.arithmeticMean(atom_counts)
        self.gld_req = Records.arithmeticMean(gld)
        self.gst_req = Records.arithmeticMean(gst)

class EdgeData:
    def __init__(self, idx, call_type, clock_cycles, num_traversals, avg_exec_time, avg_exec_time_std_dev):
        self.idx = idx
        self.call_type = call_type
        self.clock_cycles = clock_cycles
        self.num_traversals = num_traversals
        self.avg_exec_time = avg_exec_time
        self.avg_exec_time_std_dev = avg_exec_time_std_dev
        self.sg_idx = None

class EdgeDataList:
    def __init__(self, filename, num_runs):
        if not os.path.isfile(filename):
            raise Exception("file not found {0}".format(filename))
        self.edges = []
        raw_edge_data = []
        known_edges = {}
        self.map = {}
        self.clock_rate = 0
        with codecs.open(filename, "r", "utf-8") as file: 
            reader = csv.reader(file, delimiter = ';')
            for row in reader:
                if len(row) == 0:
                    continue
                if row[0] == "clock_rate":
                    self.clock_rate = int(row[1])
                elif row[0].isdigit():
                    assert len(row) == 5
                    edge_idx = int(row[0])
                    call_type = int(row[1])
                    if call_type == 0:
                        continue
                    if edge_idx in known_edges:
                        idx = known_edges.get(edge_idx)
                        raw_edge_data[idx][2].append(int(row[2]))
                    else:
                        raw_edge_data.append([edge_idx, call_type, [int(row[2])], int(row[3])])
                        known_edges[edge_idx] = len(raw_edge_data) - 1

        for r in raw_edge_data:
            self.map[r[0]] = len(self.edges)
            exec_times = [clock_cycle / (self.clock_rate * 1000.0 * r[3]) for clock_cycle in r[2]]
            avg_exec_time_std_dev = standard_deviation(exec_times)
            avg_exec_time = Records.arithmeticMean(exec_times)
            self.edges.append(EdgeData(r[0], r[1], r[2], r[3], avg_exec_time, avg_exec_time_std_dev))

    def __iter__(self):
       for x in self.edges:
          yield x

    def getEdge(self, edge_idx):
        if not edge_idx in self.map:
            return None
        idx = self.map.get(edge_idx)
        return self.edges[idx]

class SubGraphData:
    def __init__(self, idx, clock_cycles, num_traversals):
        self.idx = idx
        self.clock_cycles = clock_cycles
        self.num_traversals = num_traversals
        self.avg_exec_time = None
        self.avg_exec_time_std_dev = None

class SubGraphDataList:
    def __init__(self, filename, num_runs):
        if not os.path.isfile(filename):
            raise Exception("file not found {0}".format(filename))
        self.subgraphs = []
        raw_subgraph_data = []
        known_subgraphs = {}
        self.clock_rate = 0
        with codecs.open(filename, "r", "utf-8") as file:
            reader = csv.reader(file, delimiter = ';')
            for row in reader:
                if len(row) == 0:
                    continue
                if row[0] == "clock_rate":
                    self.clock_rate = int(row[1])
                if row[0].isdigit():
                    assert len(row) == 4
                    sg_idx = int(row[0])
                    if sg_idx in known_subgraphs:
                        idx = known_subgraphs.get(sg_idx)
                        raw_subgraph_data[idx][1].append(int(row[1]))
                    else:
                        raw_subgraph_data.append([sg_idx, [int(row[1])], int(row[2])])
                        known_subgraphs[sg_idx] = len(raw_subgraph_data) - 1

        for r in raw_subgraph_data:
            self.subgraphs.append(SubGraphData(r[0], r[1], r[2]))

    def __iter__(self):
       for x in self.subgraphs:
          yield x

class Metadata:
        def __init__(self, numRuns, minNumElements, maxNumElements, elementsStep):
            self.numRuns = numRuns
            self.minNumElements = minNumElements
            self.maxNumElements = maxNumElements
            self.elementsStep = elementsStep
            self.instrumented = False
            self.optimization = 0

        def __getstate__(self):
            return (self.numRuns, self.minNumElements, self.maxNumElements, self.elementsStep, self.instrumented, self.optimization)

        def __setstate__(self, state):
            self.numRuns = state[0]
            self.minNumElements = state[1]
            self.maxNumElements = state[2]
            self.elementsStep = state[3]
            self.instrumented = state[4]
            self.optimization = state[5]

        
        @staticmethod
        def loadFromFile(fileName):
            file = open(fileName)
            try:
                numRuns = int(file.readline())
                minNumElements = int(file.readline())
                maxNumElements = int(file.readline())
                elementsStep = int(file.readline())
            except ValueError:
                raise LoadingError("corrupted metadata file")
            file.close()
            return Metadata(numRuns, minNumElements, maxNumElements, elementsStep)

class LoadingError(Exception):
    pass

class Execution:
    def __init__(self, name, metadata, generationTime, genTimeStDev):
        self.name = name
        self.metadata = metadata
        self.numTerminalsPerElement = 1
        self.color = '#000000'
        self.generationTime = generationTime
        self.genTimeStDev = genTimeStDev
        self.cuptiData = None
        self.edgeData = None
        self.subGraphData = None
        # changed for random rule test case!!!!!!!   
        #self.num_axioms = name.split("_")[3]
        self.num_axioms = name.split("_")[2]


    @staticmethod
    def loadFromCache(cacheDir, name, expectedNumGenerationRecords = None):
        metadataFileName = "{0}\\{1}-auto.txt".format(cacheDir, name)
        generationRecordsFileName = "{0}\\{1}-generation.txt".format(cacheDir, name)
        cuptiRecordsFileName = "{0}\\{1}-cupti.txt".format(cacheDir, name)
        edgeRecordsFileName = "{0}\\{1}-edges.csv".format(cacheDir, name)
        subGraphRecordsFileName = "{0}\\{1}-subgraphs.csv".format(cacheDir, name)

        if not os.path.isfile(metadataFileName):
            raise LoadingError("cached header file not found {0}".format(metadataFileName))

        if not os.path.isfile(generationRecordsFileName):
            raise LoadingError("cached generation records file not found {0}".format(generationRecordsFileName))

        metadata = Metadata.loadFromFile(metadataFileName)
        generationRecords = Records.loadFromFileAsFloats(generationRecordsFileName)

        if not name.find("_i") == -1:
            metadata.instrumented = True
            if not os.path.isfile(cuptiRecordsFileName):
                raise LoadingError("cached cupti records file not found {0}".format(cuptiRecordsFileName))
        
            if not os.path.isfile(edgeRecordsFileName):
                raise LoadingError("cached edges records file not found {0}".format(edgeRecordsFileName))

            if not os.path.isfile(subGraphRecordsFileName):
                raise LoadingError("cached subgraph records file not found {0}".format(subGraphRecordsFileName))

        match = re.search("_o([0-3])", name)
        if match is not None:
            assert match.group(1) is not None
            optimization = int(match.group(1))
            metadata.optimization = optimization

        numGenerationRecords = len(generationRecords)
        if expectedNumGenerationRecords != None and numGenerationRecords < expectedNumGenerationRecords:
            raise LoadingError("insufficient generation records for {0} (got: {1}, expected: {2})".format(name, numGenerationRecords, expectedNumGenerationRecords))

        execution = Execution(name, metadata, Records.arithmeticMean(generationRecords), standard_deviation(generationRecords))

        if metadata.instrumented:
            execution.cuptiData = CuptiData(cuptiRecordsFileName)
            execution.edgeData = EdgeDataList(edgeRecordsFileName, metadata.numRuns)
            execution.subGraphData = SubGraphDataList(subGraphRecordsFileName, metadata.numRuns)

        return execution

    def computeSubGraphExecTime(self, conn_map):
        assert self.subGraphData.clock_rate == self.edgeData.clock_rate
        clock_rate = self.subGraphData.clock_rate
        for s in self.subGraphData:
            if not s.idx in conn_map:
                if s.num_traversals != 0:
                    sg_avg_exec_times = [clock_cycle / (clock_rate * 1000.0 * s.num_traversals) for clock_cycle in s.clock_cycles]
                    sg_avg_exec_time_std_dev = standard_deviation(sg_avg_exec_times)
                    s.avg_exec_time = Records.arithmeticMean(sg_avg_exec_times)
                    s.avg_exec_time_std_dev = sg_avg_exec_time_std_dev
                else:
                    s.avg_exec_time = 0
                    s.avg_exec_time_std_dev = 0
            else:
                conns = conn_map[s.idx]

                edges_avg_exec_times = 0
                for edge_idx in conns:
                    edge = self.edgeData.getEdge(edge_idx)
                    edges_avg_exec_times = edges_avg_exec_times + edge.avg_exec_time

                sg_compensated_clock_cycles = []
                for i, clock_cycle in enumerate(s.clock_cycles):
                    sg_compensated_clock_cycle = clock_cycle
                    for edge_idx in conns:
                        edge = self.edgeData.getEdge(edge_idx)
                        sg_compensated_clock_cycle = sg_compensated_clock_cycle - edge.clock_cycles[i]
                    sg_compensated_clock_cycles.append(sg_compensated_clock_cycle)

                sg_avg_exec_times = [sg_compensated_clock_cycle / (clock_rate * 1000.0 * s.num_traversals) for sg_compensated_clock_cycle in sg_compensated_clock_cycles]
                sg_avg_exec_time_std_dev = standard_deviation(sg_avg_exec_times)

                s.avg_exec_time = Records.arithmeticMean(sg_avg_exec_times)
                s.avg_exec_time_std_dev = sg_avg_exec_time_std_dev

                assert s.avg_exec_time > 0

    def setEdgeSources(self, conn_map):
        for s in self.subGraphData:
            if not s.idx in conn_map:
                continue
            conns = conn_map[s.idx]
            for edge_idx in conns:
                edge = self.edgeData.getEdge(edge_idx)
                edge.sg_idx = s.idx