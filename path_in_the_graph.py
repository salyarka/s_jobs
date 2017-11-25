"""Script finds the shortest path in oriented graph.
Input is a file with vertices separated by tab ('\t')in a form:
    vertex_1    vertex_2
    vertex_3    vertex_4
    ...
where the direction of the edge goes from left vertex to right vertex.
Vertices are denoted by integers.
"""

import sys

from pyspark import SparkConf, SparkContext


def parse_relation(s):
    to, source = s.split('\t')
    return int(source), int(to)


def step(item):
    prev_v, prev_d, next_v = item[0], item[1][0][0], item[1][1]
    history = item[1][0][1][:]
    history.append(next_v)
    return next_v, (prev_d + 1, history)


def complete(item):
    v, old_d, new_d = item[0], item[1][0], item[1][1]
    if old_d is None or new_d is None:
        distance = new_d if old_d is None else old_d
    else:
        distance = new_d if old_d[0] > new_d[0] else old_d
    return v, distance


def reduce_distances(a, b):
    if a[0] is None:
        return a
    return a if a[0][0] <= b[0][0] else b

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(
            'Usage: <start_vertex> <end_vertex> <file_name>'
            ' <number_of_partitions>'
        )
        exit(1)

    sc = SparkContext(conf=SparkConf().setAppName('MyApp').setMaster('local'))

    n = int(sys.argv[4])

    edges = sc.textFile(sys.argv[3]).map(parse_relation)\
                                    .partitionBy(n)\
                                    .persist()

    start_distance = 0
    start_vertex = int(sys.argv[1])
    finish_vertex = int(sys.argv[2])

    distances = sc.parallelize([
        (start_vertex, (start_distance, [start_vertex]))
    ]).partitionBy(n)

    while True:
        candidates = distances.join(edges, n).map(step)
        new_distances = distances.fullOuterJoin(candidates, n) \
                                 .reduceByKey(reduce_distances)\
                                 .map(complete, True)\
                                 .persist()
        finish = new_distances.filter(
            lambda x: finish_vertex == x[0]
        ).count() > 0
        if not finish:
            start_distance += 1
            distances = new_distances
        else:
            break

    paths = new_distances.filter(lambda x: finish_vertex == x[0]).collect()
    res = (str(x) for x in paths[0][1][1])
    print(','.join(res))
