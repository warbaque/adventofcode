import hashlib
import itertools
import re
import json


#===============================================================================
# HELPERS

class printer():
    day = ""
    def hline(self):
        print("+-----+------------+")
    def header(self):
        self.hline()
        print("| {:>3s} | {:>10s} |".format("DAY", "results"))
        self.hline()
    def row(self, solution):
        print("| {:>3s} | {:>10s} |".format(self.day, str(solution)))
        self.day = ""

printer = printer()

def solver(input_file, day):
    printer.day = input_file[6:-4]
    with open(input_file) as f:
        day(f.read())
    printer.hline()

DISABLE_TOO_SLOW = False


#===============================================================================
# SOLUTIONS
#===============================================================================


#===============================================================================
# DAY 1

def day_1(s):

    printer.row(s.count("(")-s.count(")"))

    floor = 0
    i = 0
    while (i < len(s) and floor != -1):
        floor += 1 if s[i] == "(" else -1
        i += 1
    printer.row(i)


#===============================================================================
# DAY 2

def day_2(s):

    boxes = s.strip().split("\n")

    total = 0
    for box in boxes:
        l = [int(i) for i in box.split("x")]
        a = [l[i] * l[i-1] for i in range(3)]
        total += 2 * sum(a) + min(a)
    printer.row(total)

    total = 0
    for box in boxes:
      l = [int(i) for i in box.split("x")]
      total += 2*sum(l) - 2*max(l) + l[0]*l[1]*l[2]
    printer.row(total)


#===============================================================================
# DAY 3

def day_3(s):

    def next_v(c, v):
        if   c == '<': return tuple(( v[0]-1 , v[1]   ))
        elif c == '>': return tuple(( v[0]+1 , v[1]   ))
        elif c == '^': return tuple(( v[0]   , v[1]-1 ))
        elif c == 'v': return tuple(( v[0]   , v[1]+1 ))

    visited = {}
    v = tuple((0, 0))
    visited[v] = True
    for c in s:
        v = next_v(c, v)
        visited[v] = True
    printer.row(len(visited))

    visited = {}
    v = [tuple((0, 0)), tuple((0, 0))]
    visited[v[0]] = True
    for i, c in enumerate(s):
        v[i%2] = next_v(c, v[i%2])
        visited[v[i%2]] = True
    printer.row(len(visited))


#===============================================================================
# DAY 4

def day_4(s):

    s = s.strip()

    i = 0
    while(not hashlib.md5(s+str(i)).hexdigest().startswith("00000")):
        i += 1
    printer.row(i)

    if DISABLE_TOO_SLOW: return
    while(not hashlib.md5(s+str(i)).hexdigest().startswith("000000")):
        i += 1
    printer.row(i)


#===============================================================================
# DAY 5

def day_5(s):

    def is_valid_1(string):
        num_vowels = 0
        has_repeat = False
        previous_c = ""
        for c in string:
            if ((previous_c == 'a' and c == 'b') or
                (previous_c == 'c' and c == 'd') or
                (previous_c == 'p' and c == 'q') or
                (previous_c == 'x' and c == 'y')):
                return False
            if c in "aeiou":
                num_vowels += 1
            if c == previous_c:
                has_repeat = True
            previous_c = c
        return num_vowels >= 3 and has_repeat

    def is_valid_2(string):
        has_xyx  = False
        has_xyxy = False
        c_2 = ""
        c_1 = ""
        for i, c in enumerate(string):
            if c == c_2:
                has_xyx = True
            if c_2 and (c_2+c_1 in string[i:]):
                has_xyxy = True
            c_2 = c_1
            c_1 = c
        return has_xyx and has_xyxy

    valid_strings = 0
    for string in s.split("\n"):
        if is_valid_1(string):
            valid_strings += 1
    printer.row(valid_strings)

    valid_strings = 0
    for string in s.split("\n"):
        if is_valid_2(string):
            valid_strings += 1
    printer.row(valid_strings)


#===============================================================================
# DAY 6

def day_6(s):

    cmds = s.strip().split("\n")
    prog = re.compile('(.*) (\d+),(\d+) through (\d+),(\d+)')

    array_1 = [[False for _ in range(1000)] for _ in range(1000)]
    array_2 = [[0 for _ in range(1000)] for _ in range(1000)]

    for cmd in cmds:

        m = prog.match(cmd)
        x1 = int(m.group(2))
        y1 = int(m.group(3))
        x2 = int(m.group(4))
        y2 = int(m.group(5))

        if (m.group(1) == 'toggle'):
            for x in range(x1, x2+1):
                for y in range(y1, y2+1):
                    array_1[x][y] = not array_1[x][y]
                    array_2[x][y] += 2

        elif (m.group(1) == 'turn on'):
            for x in range(x1, x2+1):
                for y in range(y1, y2+1):
                    array_1[x][y] = True
                    array_2[x][y] += 1

        elif (m.group(1) == 'turn off'):
            for x in range(x1, x2+1):
                for y in range(y1, y2+1):
                    array_1[x][y] = False
                    array_2[x][y] = max(array_2[x][y]-1, 0)

    printer.row(sum([row.count(True) for row in array_1]))
    printer.row(sum([sum(row) for row in array_2]))


#===============================================================================
# DAY 7

def day_7(s):

    gates = s.strip().split("\n")
    circuit = {}
    for gate in gates:
        func, wire = gate.split(' -> ')
        circuit[wire] = func

    def gate_not(i):
        return 65535-circuit_get(i)
    def gate_and(a, b):
        return circuit_get(a) & circuit_get(b)
    def gate_or(a, b):
        return circuit_get(a) | circuit_get(b)
    def gate_lshift(i, shift):
        return circuit_get(i) << int(shift)
    def gate_rshift(i, shift):
        return circuit_get(i) >> int(shift)

    regex = [
        (gate_not,    re.compile('NOT (\S+)')),
        (gate_and,    re.compile('(\S+) AND (\S+)')),
        (gate_or,     re.compile('(\S+) OR (\S+)')),
        (gate_lshift, re.compile('(\S+) LSHIFT (\d+)')),
        (gate_rshift, re.compile('(\S+) RSHIFT (\d+)'))
    ]

    def circuit_get(wire):

        if wire.isdigit():
            return int(wire)

        if isinstance(circuit[wire], int) or circuit[wire].isdigit():
            return int(circuit[wire])

        m = None
        for func, r in regex:
            m = r.match(circuit[wire])
            if m:
                circuit[wire] = func(*m.groups())
                return circuit[wire]

        circuit[wire] = circuit_get(circuit[wire])
        return circuit[wire]

    circuit_copy = circuit.copy()
    a = circuit_get("a")
    printer.row(a)

    circuit = circuit_copy
    circuit['b'] = a
    printer.row(circuit_get("a"))


#===============================================================================
# DAY 8

def day_8(s):

    strings = s.strip().split("\n")

    total_1 = 0
    total_2 = 0

    for string in strings:

        chars = len(string)-2
        i = 0
        while i < len(string):
            i = string.find('\\', i)
            if i == -1: break
            l = 4 if string[i+1] == 'x' else 2
            chars -= l-1
            i += l

        total_1 += len(string) - chars
        total_2 += string.count('\"') + string.count('\\') + 2

    printer.row(total_1)
    printer.row(total_2)


#===============================================================================
# DAY 9

def day_9(s):

    r = re.compile('(\S+) to (\S+) = (\d+)')

    distances = s.strip().split("\n")
    adj = {}

    for distance in distances:
        m = r.match(distance)
        loc_1, loc_2, d = m.groups()

        if not loc_1 in adj:
            adj[loc_1] = {}
        if not loc_2 in adj:
            adj[loc_2] = {}

        adj[loc_1][loc_2] = int(d)
        adj[loc_2][loc_1] = int(d)

    min_dist = float("inf")
    max_dist = 0

    for p in itertools.permutations(adj.keys()):
        dist = 0
        for i in range(len(p)-1):
            dist += adj[p[i]][p[i+1]]

        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

    printer.row(min_dist)
    printer.row(max_dist)


#===============================================================================
# DAY 10

def day_10(s):

    s = s.strip()

    def next_s(s):
        new_s = ""
        for k, g in itertools.groupby(s):
            new_s += str(len(list(g))) + k
        return new_s

    for _ in range(40):
        s = next_s(s)
    printer.row(len(s))

    if DISABLE_TOO_SLOW: return
    for _ in range(10):
        s = next_s(s)
    printer.row(len(s))


#===============================================================================
# DAY 11

def day_11(s):

    s = s.strip()

    def increment(s):
        c = s[-1]
        s = s[:-1]
        if c == 'z':
            s = increment(s)
            c = 'a'
        else:
            c = chr(ord(c) + 1)
        return s + c

    def is_valid(s):

        pairs = []
        cur_straight = 1
        max_straight = 0

        prev_c = ' '
        for c in s:
            if c in 'iol': return False
            if c == prev_c and not c in pairs: pairs.append(c)
            if ord(c) == ord(prev_c)+1:
                cur_straight += 1
                max_straight = max(cur_straight, max_straight)
            else:
                cur_straight = 1
            prev_c = c
        return len(pairs) > 1 and max_straight > 2

    while(not is_valid(s)):
        s = increment(s)
    printer.row(s)

    s = increment(s)
    while(not is_valid(s)):
        s = increment(s)
    printer.row(s)


#===============================================================================
# DAY 12

def day_12(s):

    printer.row(sum(map(int, re.findall('-?\d+', s))))

    def recur(c):
        if isinstance(c, list):
            return sum((recur(i) for i in c))
        elif isinstance(c, dict):
            return 0 if "red" in c.values() else sum((recur(c[k]) for k in c))
        return c if isinstance(c, int) else 0

    printer.row(recur(json.loads(s)))


#===============================================================================
# DAY 13

def day_13(s):

    r = re.compile('(\S+) would (gain|lose) (\d+) happiness units by sitting next to (\S+).')

    preferences = s.strip().split("\n")
    happiness   = {}

    for pref in preferences:
        m = r.match(pref)

        u = m.group(1)
        v = m.group(4)
        dif = (1 if m.group(2) == 'gain' else -1) * int(m.group(3))

        if not u in happiness:
            happiness[u] = {}
        happiness[u][v] = dif

    def max_happiness():
        max_happiness = float("-inf")
        for p in itertools.permutations(happiness.keys()[1:]):

            p = (happiness.keys()[0],) + p
            happy = sum((happiness[k][p[i-1]] + happiness[k][p[i-len(p)+1]]
                for i, k in enumerate(p)))
            max_happiness = max(max_happiness, happy)

        return max_happiness

    printer.row(max_happiness())

    happiness['me'] = {}
    for u in happiness:
        if u == 'me': continue
        happiness[u]['me'] = 0
        happiness['me'][u] = 0

    printer.row(max_happiness())


#===============================================================================
# DAY 14

def day_14(s):

    t_total = 2503
    r = re.compile('\S+ can fly (\d+) km/s for (\d+) seconds, but then must rest for (\d+) seconds.')
    SPEED, T_FLYING, T_RESTING, DIST, POINTS = range(5)

    reindeer_stats = s.strip().split("\n")
    reindeers = [list(map(int, r.match(reindeer).groups())) + [0, 0] for reindeer in reindeer_stats]

    max_dist   = 0
    max_points = 0

    for t in range(t_total):
        for reindeer in reindeers:
            is_flying = t % (reindeer[T_FLYING] + reindeer[T_RESTING]) < reindeer[T_FLYING]
            if is_flying:
                reindeer[DIST] += reindeer[SPEED]
                max_dist = max(reindeer[DIST], max_dist)
        for reindeer in reindeers:
            if reindeer[DIST] == max_dist:
                reindeer[POINTS] += 1
                max_points = max(reindeer[POINTS], max_points)

    printer.row(max_dist)
    printer.row(max_points)

#===============================================================================

DISABLE_TOO_SLOW = False

printer.header()
solver("input/1.txt", day_1)
solver("input/2.txt", day_2)
solver("input/3.txt", day_3)
solver("input/4.txt", day_4)
solver("input/5.txt", day_5)
solver("input/6.txt", day_6)
solver("input/7.txt", day_7)
solver("input/8.txt", day_8)
solver("input/9.txt", day_9)
solver("input/10.txt", day_10)
solver("input/11.txt", day_11)
solver("input/12.txt", day_12)
solver("input/13.txt", day_13)
solver("input/14.txt", day_14)
