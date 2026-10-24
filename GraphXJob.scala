// GraphXJob.scala
// ================
// Milestone 4 — Native Spark GraphX Analysis
//
// Run with:
//   JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home \
//   /opt/homebrew/Cellar/apache-spark/4.2.0/libexec/bin/spark-shell \
//     --driver-memory 4g \
//     --conf spark.hadoop.fs.defaultFS=hdfs://localhost:9000 \
//     -i src/GraphXJob.scala
//
// Reads:  hdfs://localhost:9000/graphrag/edges/   (src_id LONG, dst_id LONG, similarity FLOAT)
// Writes: hdfs://localhost:9000/graphrag/graphx_out/  (passage_idx LONG, pagerank DOUBLE,
//           component_id LONG, triangle_count INT, degree INT)
// Also writes local CSV: data/graphx_results.csv

import org.apache.spark.graphx._
import org.apache.spark.graphx.lib.{PageRank, ConnectedComponents, TriangleCount}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SaveMode}
import org.apache.spark.sql.types._

// ── Config ──────────────────────────────────────────────────────────────────
val EDGES_PATH    = "hdfs://localhost:9000/graphrag/edges"
val OUT_PATH      = "hdfs://localhost:9000/graphrag/graphx_out"
val LOCAL_CSV     = "data/graphx_results.csv"
val PR_ITERATIONS = 20
val PR_RESET_PROB = 0.15   // damping = 0.85

println("\n=== M4: Spark GraphX Analysis ===")

// ── Load edges from HDFS ────────────────────────────────────────────────────
println(s"Loading edges from $EDGES_PATH")
val edgesDF = spark.read.parquet(EDGES_PATH)
edgesDF.printSchema()
val nEdgesRaw = edgesDF.count()
println(s"  Raw edge rows: $nEdgesRaw")

// ── Build GraphX Edge RDD ────────────────────────────────────────────────────
// Edge attribute = similarity (Float cast to Double for GraphX)
val edgeRDD: RDD[Edge[Double]] = edgesDF.rdd.flatMap { row =>
  val src = row.getLong(0)
  val dst = row.getLong(1)
  val sim = row.getFloat(2).toDouble
  // Undirected: add both directions so all algorithms work correctly
  Seq(Edge(src, dst, sim), Edge(dst, src, sim))
}

// Vertex RDD: just the IDs, attribute = the ID itself (Long)
val vertexRDD: RDD[(VertexId, Long)] = edgeRDD.flatMap(e => Seq(e.srcId, e.dstId))
  .distinct()
  .map(id => (id, id))

val graph: Graph[Long, Double] = Graph(vertexRDD, edgeRDD)
  .partitionBy(PartitionStrategy.EdgePartition2D)
  .cache()

val nVertices = graph.vertices.count()
val nEdges    = graph.edges.count()
println(s"  Graph: $nVertices vertices, $nEdges directed edges")

// ── PageRank ─────────────────────────────────────────────────────────────────
println(s"Running PageRank (iterations=$PR_ITERATIONS, resetProb=$PR_RESET_PROB) …")
val prGraph = PageRank.run(graph, PR_ITERATIONS, PR_RESET_PROB)
val pageranks: RDD[(VertexId, Double)] = prGraph.vertices
println("  PageRank done.")

// ── Connected Components ─────────────────────────────────────────────────────
println("Running ConnectedComponents …")
val ccGraph = ConnectedComponents.run(graph)
val components: RDD[(VertexId, Long)] = ccGraph.vertices  // value = min vertex in component
println("  ConnectedComponents done.")

val nComponents = components.values.distinct().count()
println(s"  Distinct components: $nComponents")

// ── Triangle Count ───────────────────────────────────────────────────────────
println("Running TriangleCount …")
// TriangleCount requires undirected canonical graph (src < dst, no duplicate edges)
val canonicalEdges: RDD[Edge[Double]] = edgesDF.rdd.map { row =>
  val src = row.getLong(0)
  val dst = row.getLong(1)
  val sim = row.getFloat(2).toDouble
  if (src < dst) Edge(src, dst, sim) else Edge(dst, src, sim)
}.distinct()

val triGraph = Graph(vertexRDD, canonicalEdges)
  .partitionBy(PartitionStrategy.EdgePartition2D)

val tcGraph = TriangleCount.run(triGraph)
val triangles: RDD[(VertexId, Int)] = tcGraph.vertices.map { case (id, cnt) => (id, cnt) }
println("  TriangleCount done.")

// ── Degree ───────────────────────────────────────────────────────────────────
println("Computing degree …")
val degrees: RDD[(VertexId, Int)] = graph.degrees
println("  Degree done.")

// ── Join all metrics per vertex ───────────────────────────────────────────────
println("Joining all metrics …")
val joined: RDD[(VertexId, (Double, Long, Int, Int))] =
  pageranks
    .join(components)
    .join(triangles)
    .join(degrees)
    .map { case (vid, (((pr, cc), tri), deg)) => (vid, (pr, cc, tri, deg)) }

val nJoined = joined.count()
println(s"  Joined rows: $nJoined (expect $nVertices)")

// ── Write to HDFS as Parquet ─────────────────────────────────────────────────
println(s"Writing results to $OUT_PATH …")

val schema = StructType(Seq(
  StructField("passage_idx",    LongType,   nullable = false),
  StructField("pagerank",       DoubleType, nullable = false),
  StructField("component_id",   LongType,   nullable = false),
  StructField("triangle_count", IntegerType,nullable = false),
  StructField("degree",         IntegerType,nullable = false),
))

val rowsRDD: RDD[Row] = joined.map { case (vid, (pr, cc, tri, deg)) =>
  Row(vid, pr, cc, tri, deg)
}

val resultDF = spark.createDataFrame(rowsRDD, schema)
resultDF.write.mode(SaveMode.Overwrite).parquet(OUT_PATH)
println("  Write complete.")

// ── Verify from HDFS ─────────────────────────────────────────────────────────
println(s"Verifying $OUT_PATH …")
val verifyDF = spark.read.parquet(OUT_PATH)
verifyDF.printSchema()
val nVerify = verifyDF.count()
println(s"  Rows in HDFS: $nVerify")

println("  Sample rows (ordered by pagerank desc):")
verifyDF.orderBy(org.apache.spark.sql.functions.desc("pagerank")).show(10, truncate = false)

// ── Write local CSV (for dashboard and M5) ───────────────────────────────────
println(s"Writing local CSV: $LOCAL_CSV")
val collected = verifyDF.orderBy("passage_idx").collect()

// Normalise PageRank for use in scoring (0-1 range)
val prValues = collected.map(_.getDouble(1))
val prMin    = prValues.min
val prMax    = prValues.max
val prRange  = if (prMax > prMin) prMax - prMin else 1.0

val fw = new java.io.FileWriter(LOCAL_CSV)
fw.write("passage_idx,pagerank,norm_pagerank,component_id,triangle_count,degree\n")
collected.foreach { row =>
  val idx    = row.getLong(0)
  val pr     = row.getDouble(1)
  val normPr = (pr - prMin) / prRange
  val cc     = row.getLong(2)
  val tri    = row.getInt(3)
  val deg    = row.getInt(4)
  fw.write(s"$idx,$pr,$normPr,$cc,$tri,$deg\n")
}
fw.close()

val csvSize = new java.io.File(LOCAL_CSV).length()
println(s"  $LOCAL_CSV: ${csvSize / 1024} KB, ${collected.length} rows")

// ── Summary ───────────────────────────────────────────────────────────────────
val prStats = verifyDF.select(
  org.apache.spark.sql.functions.min("pagerank"),
  org.apache.spark.sql.functions.max("pagerank"),
  org.apache.spark.sql.functions.avg("pagerank")
).collect().head

println("\n=== GRAPHX SUMMARY ===")
println(s"  Vertices        : $nVertices")
println(s"  Directed edges  : $nEdges")
println(s"  Components      : $nComponents")
println(s"  PageRank min    : ${prStats.getDouble(0)}")
println(s"  PageRank max    : ${prStats.getDouble(1)}")
println(s"  PageRank avg    : ${prStats.getDouble(2)}")
println(s"  HDFS output     : $OUT_PATH")
println(s"  Local CSV       : $LOCAL_CSV")
println(s"\nMilestone 4 ${if (nVerify == nVertices) "PASSED ✓" else "FAILED ✗"}")

System.exit(0)
