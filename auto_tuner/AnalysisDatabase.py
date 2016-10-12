import os
import sqlite3

#######################
#   TRANSFER OBJECTS
#######################

####################################################################################################
class Partition:
    def __init__(self):
        self.pk = None
        self.idx = None
        self.uid = None
        self.num_edges = None
        self.instr = None
        self.opt = None
        self.exec_time = None
        self.num_axioms = None
        self.stdev = None

####################################################################################################
class Subgraph:
    def __init__(self):
        self.pk = None
        self.partition_pk = None
        self.idx = None
        self.freq = None
        self.exec_time = None
        self.stdev = None

####################################################################################################
class Edge:
    def __init__(self):
        self.pk = None
        self.partition_pk = None
        self.idx = None
        self.type = None
        self.freq = None
        self.exec_time = None
        self.subgraph_pk = None
        self.stdev = None

###########
#   MISC
###########

####################################################################################################
def assembleWhereClause(fields, prefix = ""):
    return str.join(" and ", [(prefix + key) + " = " + fieldToStr(fields[key]) for key in iter(fields)])

####################################################################################################
def fieldToStr(field):
    if field == None:
        return "null"
    elif type(field) is str:
        return "'" + field + "'"
    elif type(field) is bool:
        if field == True:
            return "1"
        else:
            return "0"
    else:
        return str(field)

####################################################################################################
def intToBool(int):
    if int == 0:
        return False
    else:
        return True

####################################################################################################
class Database:
    connection = None

    @staticmethod
    def open(name):
        Database.connection = sqlite3.connect(name + ".db")
        Database.connection.row_factory = sqlite3.Row
        Database.connection.execute('pragma foreign_keys=ON;')
        foreign_keys = Database.connection.execute('pragma foreign_keys;').fetchone()[0]
        if foreign_keys != 1:
            raise Exception("foreign_keys != 1")
        # autocommit by default
        Database.connection.isolation_level = None

    @staticmethod
    def execute(sql, params):
        if Database.connection == None:
            raise Exception("Database.connection == None")

        rowcount = Database.connection.execute(sql, params).rowcount
        return rowcount

    @staticmethod
    def executeAndReturnLastRowId(sql, params):
        if Database.connection == None:
            raise Exception("Database.connection == None")

        lastrowid = Database.connection.execute(sql, params).lastrowid
        return lastrowid

    @staticmethod
    def fetch(sql):
        if Database.connection == None:
            raise Exception("Database.connection == None")

        cursor = Database.connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    @staticmethod
    def fetchOne(sql):
        if Database.connection == None:
            raise Exception("connection == None")

        cursor = Database.connection.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        cursor.close()
        return row

    @staticmethod
    def __create(name, script_filename):
        Database.open(name)
        if script_filename != None:
            if not os.path.exists(script_filename):
                raise Exception("script \"{0}\" not found".format(script_filename))
            with open(script_filename, "r") as script_file:
                script = script_file.read()
            Database.connection.executescript(script)
        Database.close()

    @staticmethod
    def createIfNeeded(name, script_filename=None):
        if os.path.exists(name + ".db"):
            return
        Database.__create(name, script_filename)

    @staticmethod
    def create(name, script_filename=None):
        if os.path.exists(name + ".db"):
            os.remove(name + ".db")
        Database.__create(name, script_filename)

    @staticmethod
    def close():
        if Database.connection != None:
            Database.connection.close()

##########################
#   DATA ACCESS OBJECTS
##########################

####################################################################################################
class PartitionDAO:
    @staticmethod
    def insert(partition):
        if partition.pk != None:
            raise Exception("partition.pk != None")
        lastrowid = Database.executeAndReturnLastRowId("insert into partition (idx, uid, num_edges, instr, opt, exec_time, stdev, num_axioms) values (?, ?, ?, ?, ?, ?, ?, ?);", [partition.idx, partition.uid, partition.num_edges, partition.instr, partition.opt, partition.exec_time, partition.stdev, partition.num_axioms])
        partition.pk = lastrowid

    @staticmethod
    def update(partition):
        if partition.pk == None:
            raise Exception("partition.pk != None")
        rowcount = Database.execute("update partition set idx=?, uid=?, num_edges=?, instr=?, opt=?, exec_time=?, num_axioms=?, stdev=?, where pk=?;", [partition.idx, partition.uid, partition.num_edges, partition.instr, partition.opt, partition.exec_time, partition.num_axioms, partition.stdev, partition.pk])
        if rowcount == 0:
            raise Exception("rowcount == 0")

    @staticmethod
    def __fromRow(row):
        if row == None:
            raise Exception("row == None")
        result = Partition()
        result.pk = row["pk"]
        result.idx = row["idx"]
        result.uid = row["uid"]
        result.num_edges = row["num_edges"]
        result.instr = intToBool(row["instr"])
        result.opt = row["opt"]
        result.exec_time = row["exec_time"]
        result.stdev = row["stdev"]
        result.num_axioms = row["num_axioms"]
        return result

    @staticmethod
    def list():
        rows = Database.fetch("select pk, idx, uid, num_edges, instr, opt, exec_time, num_axioms, stdev from partition;")
        results = []
        for row in rows:
            results.append(PartitionDAO.__fromRow(row))
        return results

    @staticmethod
    def find(fields):
        if len(fields) == 0:
            raise Exception("len(fields) == 0")
        return PartitionDAO.__fromRow(Database.fetchOne("select pk, idx, uid, num_edges, instr, opt, exec_time, stdev, num_axioms from partition where " + assembleWhereClause(fields) + ";"))

    @staticmethod
    def count():
        return Database.fetchOne("select count(*) from partition;")[0]

    @staticmethod
    def lightWeightList():
        rows = Database.fetch("select idx, uid, num_axioms from partition;")
        results = []
        for row in rows:
            partition = Partition()
            partition.idx = row["idx"]
            partition.uid = row["uid"]
            partition.num_axioms= row["num_axioms"]
            results.append(partition)
        return results

    @staticmethod
    def globalMinMaxNumAxioms(fields):
        if len(fields) == 0:
            raise Exception("len(fields) == 0")
        row = Database.fetchOne("select min(num_axioms) as min, max(num_axioms) as max from (select num_axioms from partition where " + assembleWhereClause(fields) + ");")
        return (row["min"], row["max"])

    @staticmethod
    def execTimes(where, orderBy = ["instr", "opt"]):
        if len(where) == 0:
            raise Exception("len(fields) == 0")
        rows = Database.fetch("select exec_time from partition where " + assembleWhereClause(where) + " order by " + str.join(", ", orderBy) + ";")
        results = []
        for row in rows:
            results.append(row["exec_time"])
        return results

####################################################################################################
class SubgraphDAO:
    @staticmethod
    def insert(subgraph):
        if subgraph.pk != None:
            raise Exception("subgraph.pk != None")
        lastrowid = Database.executeAndReturnLastRowId("insert into subgraph (partition_pk, idx, freq, exec_time, stdev) values (?, ?, ?, ?, ?);", [subgraph.partition_pk, subgraph.idx, subgraph.freq, subgraph.exec_time, subgraph.stdev])
        subgraph.pk = lastrowid

    @staticmethod
    def update(subgraph):
        if subgraph.pk == None:
            raise Exception("subgraph.pk != None")
        rowcount = Database.execute("update subgraph set idx=?, freq=?, exec_time=?, stdev=? where pk=?;", [subgraph.idx, subgraph.freq, subgraph.exec_time, subgraph.stdev, subgraph.pk])
        if rowcount == 0:
            raise Exception("rowcount == 0")

    @staticmethod
    def __fromRow(row):
        if row == None:
            raise Exception("row == None")
        result = Subgraph()
        result.pk = row["pk"]
        result.partition_pk = row["partition_pk"]
        result.idx = row["idx"]
        result.freq = row["freq"]
        result.exec_time = row["exec_time"]
        result.stdev = row["stdev"]
        return result

    @staticmethod
    def list():
        rows = Database.fetch("select pk, partition_pk, idx, freq, exec_time, stdev from subgraph order by partition_pk, idx;")
        results = []
        for row in rows:
            results.append(SubgraphDAO.__fromRow(row))
        return results

    @staticmethod
    def find(fields):
        if len(fields) == 0:
            raise Exception("len(fields) == 0")
        return SubgraphDAO.__fromRow(Database.fetchOne("select pk, partition_pk, idx, freq, exec_time, stdev from subgraph from subgraph where " + assembleWhereClause(fields) + " order by partition_pk, idx;"))

    @staticmethod
    def fromPartition(fields):
        rows = Database.fetch("select pk, partition_pk, idx, freq, exec_time, stdev from subgraph where partition_pk = (select pk from partition where " + assembleWhereClause(fields) + ") order by partition_pk, idx;")
        results = []
        for row in rows:
            results.append(SubgraphDAO.__fromRow(row))
        return results

    @staticmethod
    def count():
        return Database.fetchOne("select count(*) from subgraph;")[0]
    
####################################################################################################
class EdgeDAO:
    @staticmethod
    def insert(edge):
        if edge.pk != None:
            raise Exception("edge.pk != None")
        rowcount = Database.execute("insert into edge (partition_pk, idx, freq, exec_time, stdev, type, subgraph_pk) values (?, ?, ?, ?, ?, ?, ?);", [edge.partition_pk, edge.idx, edge.freq, edge.exec_time, edge.stdev, edge.type, edge.subgraph_pk])
        if rowcount == 0:
            raise Exception("rowcount == 0")

    @staticmethod
    def update(edge):
        if edge.pk == None:
            raise Exception("edge.pk != None")
        lastrowid = Database.executeAndReturnLastRowId("update edge set idx=?, freq=?, exec_time=?, stdev=?, type=?, subgraph_pk = ? where pk=?;", [edge.idx, edge.freq, edge.exec_time, edge.stdev, edge.type, edge.subgraph_pk, edge.pk])
        edge.pk = lastrowid

    @staticmethod
    def __fromRow(row):
        if row == None:
            raise Exception("row == None")
        result = Edge()
        result.pk = row["pk"]
        result.partition_pk = row["partition_pk"]
        result.idx = row["idx"]
        result.freq = row["freq"]
        result.exec_time = row["exec_time"]
        result.stdev = row["stdev"]
        result.type = row["type"]
        result.subgraph_pk = row["subgraph_pk"]
        return result

    @staticmethod
    def list():
        rows = Database.fetch("select pk, partition_pk, idx, freq, exec_time, stdev, type, subgraph_pk from edge order by partition_pk, idx;")
        results = []
        for row in rows:
            results.append(EdgeDAO.__fromRow(row))
        return results

    @staticmethod
    def find(fields):
        if len(fields) == 0:
            raise Exception("len(fields) == 0")
        return EdgeDAO.__fromRow(Database.fetchOne("select pk, partition_pk, idx, freq, exec_time, stdev, type, subgraph_pk from edge from edge where " + assembleWhereClause(fields) + " order by partition_pk, idx;"))

    @staticmethod
    def fromPartition(fields):
        rows = Database.fetch("select pk, partition_pk, idx, freq, exec_time, stdev, type, subgraph_pk from edge where partition_pk = (select pk from partition where " + assembleWhereClause(fields) + ") order by partition_pk, idx;")
        results = []
        for row in rows:
            results.append(EdgeDAO.__fromRow(row))
        return results

    @staticmethod
    def count():
        return Database.fetchOne("select count(*) from edge;")[0]
