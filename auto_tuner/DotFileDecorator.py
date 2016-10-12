import os
import glob
import re
from bisect import bisect_left
from AnalysisDatabase import *

def binary_search(a, x, lo = 0, hi = None):
    hi = hi if hi is not None else len(a)
    pos = bisect_left(a, x, lo, hi)
    return (pos if pos != hi and a[pos] == x else -1)

def format_exec_time(exec_time):
    return '{0:.10f}'.format(exec_time)

def compare_timings(t0, t1):
    if t0 > t1:
        if t0 == 0:
            assert t0 == t1
            return ("even", "even")
        else:
            perc = (1.0 - (t1 / t0)) * 100.0
            return ("{0:.2f}% slower".format(perc), "{0:.2f}% faster".format(perc))
    else:
        if t1 == 0:
            assert t0 == t1
            return ("even", "even")
        else:
            perc = (1.0 - (t0 / t1)) * 100.0
            return ("{0:.2f}% faster".format(perc), "{0:.2f}% slower".format(perc))

def decorate_dot_file(partition_uid):
    if partition_uid == None:
        raise Exception("partitionUid == None")

    dot_files_names = glob.glob(PARTITIONS_DIR + "partition_*_" + partition_uid + "*_i.dot")
    if len(dot_files_names) != 2:
        print("WARNING: dot files not found for instrumented and optimized partition '" + partition_uid + "'")
        return False

    global_min_max = PartitionDAO.globalMinMaxNumAxioms({ "uid" : partition_uid, "instr": True })

    for k in [0, 1]:
        c1 = { "uid": partition_uid, "num_axioms": global_min_max[k], "instr": True, "opt": False }
        c2 = { "uid": partition_uid, "num_axioms": global_min_max[k], "instr": True, "opt": True }

        total_exec_times = PartitionDAO.execTimes({ "uid" : partition_uid, "num_axioms": global_min_max[k], "instr": True })

        edges = []
        edges.append(EdgeDAO.fromPartition(c1))
        edges.append(EdgeDAO.fromPartition(c2))

        edges_idxs = []
        for i, e1 in enumerate(edges[0]):
            assert e1.idx == edges[1][i].idx
            edges_idxs.append(e1.idx)

        subgraphs = []
        subgraphs.append(SubgraphDAO.fromPartition(c1))
        subgraphs.append(SubgraphDAO.fromPartition(c2))

        assert len(subgraphs[0]) > 0
        assert len(subgraphs[0]) == len(subgraphs[1])

        subgraphs_idxs = []
        for i, s1 in enumerate(subgraphs[0]):
            assert s1.idx == subgraphs[1][i].idx
            subgraphs_idxs.append(s1.idx)

        c_edges_idxs = []
        edge_cmp_strs = {}
        for edge_idx, c in enumerate(partition_uid):
            if c == "0":
                continue
            pos = binary_search(edges_idxs, edge_idx)
            if pos != -1:
                edge_cmp_strs[edge_idx] = compare_timings(edges[0][pos].exec_time, edges[1][pos].exec_time)
            else:
                c_edges_idxs.append(edge_idx)

        dot_file_content = []
        with open(dot_files_names[0], "r") as dot_file:
            dot_file_content.append(dot_file.read())

        with open(dot_files_names[1], "r") as dot_file:
            dot_file_content.append(dot_file.read())

        sgs_cmp_strs = {}
        matches = re.findall("subgraph cluster_([0-9]+) {", dot_file_content[0]);
        for match in matches:
            # FIXME: checking invariants
            assert len(match) == 1
            sg_idx = int(match[0])
            pos = binary_search(subgraphs_idxs, sg_idx)
            if pos != -1:
                sgs_cmp_strs[sg_idx] = compare_timings(subgraphs[0][pos].exec_time, subgraphs[1][pos].exec_time)
            else:
                cluster_head = "cluster_" + str(sg_idx) + " {\nlabel=\"subgraph_" + str(sg_idx)
                modified_cluster_head = cluster_head + ":\\nfreq=0\\nexec_time=0"
                dot_file_content[0] = dot_file_content[0].replace(cluster_head, modified_cluster_head)
                dot_file_content[1] = dot_file_content[1].replace(cluster_head, modified_cluster_head)

        for c_e in c_edges_idxs:
            dot_file_content[0] = re.sub("(.+) -> (.+)\[label=\"" + str(c_e) + ":", "\\1 -> \\2[label=\"edge " + str(c_e) + ":\\nfreq=0\\nexec_time=0\\nparam=", dot_file_content[0])
            dot_file_content[1] = re.sub("(.+) -> (.+)\[label=\"" + str(c_e) + ":", "\\1 -> \\2[label=\"edge " + str(c_e) + ":\\nfreq=0\\nexec_time=0\\nparam=", dot_file_content[1])

        edges_exec_time_sums = [0, 0]
        sgs_exec_time_sums = [0, 0]
        for i in [0, 1]:
            for e in edges[i]:
                edge_exec_time = e.exec_time
                edges_exec_time_sums[i] = edges_exec_time_sums[i] + edge_exec_time
                dot_file_content[i] = re.sub("(.+) -> (.+)\[label=\"" + str(e.idx) + ":", "\\1 -> \\2[label=\"edge " + str(e.idx) + ":\\nfreq=" + str(e.freq) + "\\nexec_time=" + format_exec_time(edge_exec_time) + " (" + edge_cmp_strs[e.idx][i] + ")\\nparam=", dot_file_content[i])

            for s in subgraphs[i]:
                sg_exec_time = s.exec_time
                sgs_exec_time_sums[i] = sgs_exec_time_sums[i] + sg_exec_time
                cluster_head = "cluster_" + str(s.idx) + " {\nlabel=\"subgraph_" + str(s.idx)
                modified_cluster_head = cluster_head + ":\\nfreq=" + str(s.freq) + "\\nexec_time=" + format_exec_time(s.exec_time) + " (" + sgs_cmp_strs[s.idx][i] + ")"
                dot_file_content[i] = dot_file_content[i].replace(cluster_head, modified_cluster_head)

        totals_cmp_strs = []
        totals_cmp_strs.append(compare_timings(total_exec_times[0], total_exec_times[1]))
        totals_cmp_strs.append(compare_timings(edges_exec_time_sums[0], edges_exec_time_sums[1]))
        totals_cmp_strs.append(compare_timings(sgs_exec_time_sums[0], sgs_exec_time_sums[1]))

        for i in [0, 1]:
            dot_file_content[i] = dot_file_content[i].replace("digraph operator_graph {\n", "digraph operator_graph {\nlabel=\"exec_time (total)=" + format_exec_time(total_exec_times[i]) + " (" + totals_cmp_strs[0][i] + ")\\nexec_time (edges)=" + format_exec_time(edges_exec_time_sums[i]) + " (" + totals_cmp_strs[1][i] + ")\\nexec_time (subgraphs)=" + format_exec_time(sgs_exec_time_sums[i]) + " (" + totals_cmp_strs[2][i] + ")\\nexec_time (edges + subgraphs)=" + format_exec_time(edges_exec_time_sums[i] + sgs_exec_time_sums[i]) + "\";\n")

            with open(dot_files_names[i].replace("\\partition_", "\\d_partition_" + str(global_min_max[k]) + "_"), "w") as dot_file:
                dot_file.write(dot_file_content[i])

    return True

##################################################
##################################################
##################################################

DOT_FILE_CONTENT = """digraph operator_graph {
subgraph cluster_0 {
A0[label="0[0]: IfSizeLess(X, 0.1)"]
B0[label="1[1]: Translate(1, 0, 0)"]
D0[label="3[2]: Translate(2, 0, 0)"]
}

subgraph cluster_1 {
C0[label="2[0]: Generate"]
}

A0 -> B0[label="0:  - "];
B0 -> C0[label="1:  - "];
A0 -> D0[label="2:  - "];
D0 -> C0[label="3:  - "];


}"""

# import DatabaseTest

# def test():
    # DatabaseTest.createTestBase("dot_file_decorator")
    # Database.open("dot_file_decorator")
    # with open(PARTITIONS_DIR + "partition_0_0101_i.dot", "w") as file:
        # file.write(DOT_FILE_CONTENT)
    # with open(PARTITIONS_DIR + "partition_0_0101_o_i.dot", "w") as file:
        # file.write(DOT_FILE_CONTENT)
    # decorateDotFile("0101")
    # Database.close()
    # assert os.path.exists(PARTITIONS_DIR + "d_partition_0_0101_i.dot")
    # assert os.path.exists(PARTITIONS_DIR + "d_partition_0_0101_o_i.dot")
    # os.remove(PARTITIONS_DIR + "partition_0_0101_i.dot")
    # os.remove(PARTITIONS_DIR + "partition_0_0101_o_i.dot")
    # os.remove(PARTITIONS_DIR + "d_partition_0_0101_i.dot")
    # os.remove(PARTITIONS_DIR + "d_partition_0_0101_o_i.dot")
    # os.remove("dot_file_decorator.db")

# if __name__ == "__main__":
    # test()

