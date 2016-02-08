import hashlib
import itertools
import re
import json
from math import sqrt, exp, log, ceil

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
    while(not hashlib.md5((s+str(i)).encode()).hexdigest().startswith("00000")):
        i += 1
    printer.row(i)

    if DISABLE_TOO_SLOW: return
    while(not hashlib.md5((s+str(i)).encode()).hexdigest().startswith("000000")):
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
    prog = re.compile(r'(.*) (\d+),(\d+) through (\d+),(\d+)')

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
        (gate_not,    re.compile(r'NOT (\S+)')),
        (gate_and,    re.compile(r'(\S+) AND (\S+)')),
        (gate_or,     re.compile(r'(\S+) OR (\S+)')),
        (gate_lshift, re.compile(r'(\S+) LSHIFT (\d+)')),
        (gate_rshift, re.compile(r'(\S+) RSHIFT (\d+)'))
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

    r = re.compile(r'(\S+) to (\S+) = (\d+)')

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

    printer.row(sum(map(int, re.findall(r'-?\d+', s))))

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

    r = re.compile(r'(\S+) would (gain|lose) (\d+) happiness units by sitting next to (\S+).')

    preferences = s.strip().split("\n")
    happiness   = {}

    for pref in preferences:
        m = r.match(pref)

        u = m.group(1)
        v = m.group(4)
        diff = (1 if m.group(2) == 'gain' else -1) * int(m.group(3))

        if not u in happiness:
            happiness[u] = {}
        happiness[u][v] = diff

    def max_happiness():

        max_happiness = float("-inf")
        for p in itertools.permutations(list(happiness.keys())[1:]):

            p = (list(happiness.keys())[0],) + p
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
    r = re.compile(r'\S+ can fly (\d+) km/s for (\d+) seconds, but then must rest for (\d+) seconds.')
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
# DAY 15

def day_15(s):

    spoons = 100
    ingredient_specs = s.strip().split("\n")
    ingredients = [list(map(int, re.findall(r'-?\d+', i))) for i in ingredient_specs]

    def partitions(n, k):
        for c in itertools.combinations(range(n+k-1), k-1):
            yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

    max_total = 0
    max_total_500 = 0

    for p in partitions(spoons, len(ingredients)):

        g = ([p[i] * prop for prop in ingredients[i]] for i in range(len(ingredients)))
        sums = [sum(x) for x in zip(*g)]

        total = 1
        for s in sums[:-1]:
            total *= max(s, 0)

        max_total = max(total, max_total)
        if sums[-1] == 500:
            max_total_500 = max(total, max_total_500)

    printer.row(max_total)
    printer.row(max_total_500)


#===============================================================================
# DAY 16

def day_16(s):

    sue = {
        'children': 3,
        'cats': 7,
        'samoyeds': 2,
        'pomeranians': 3,
        'akitas': 0,
        'vizslas': 0,
        'goldfish': 5,
        'trees': 3,
        'cars': 2,
        'perfumes': 1
        }

    r1 = re.compile(r'Sue (\d+): (.*)')
    r2 = re.compile(r'(\w+): (\d+)')

    aunts = s.strip().split("\n")
    aunt_1 = 0
    aunt_2 = 0

    for aunt in aunts:

        m = r1.match(aunt)
        correct_1 = True
        correct_2 = True

        for k, v in r2.findall(m.group(2)):
            if sue[k] != int(v):
                correct_1 = False

            if k in ("cats", "trees"):
                if not int(v) > sue[k]:
                    correct_2 = False
            elif k in ("pomeranians", "goldfish"):
                if not int(v) < sue[k]:
                    correct_2 = False
            elif int(v) != sue[k]:
                correct_2 = False

        if correct_1:
            aunt_1 = int(m.group(1))
        if correct_2:
            aunt_2 = int(m.group(1))

    printer.row(aunt_1)
    printer.row(aunt_2)


#===============================================================================
# DAY 17

def day_17(s):

    containers = list(map(int, s.strip().split("\n")))
    target = 150

    counts = [1] + [0]*target
    for cur_num in containers:
        for next_num in range(target, cur_num-1, -1):
            counts[next_num] += counts[next_num - cur_num]
    printer.row(counts[target])

    total = 0
    for k in range(1, len(containers) + 1):
        total = len([1 for c in itertools.combinations(containers, k) if sum(c) == target])
        if total:
            break
    printer.row(total)


#===============================================================================
# DAY 18

def day_18(s):

    light_rows = s.strip().split('\n')

    def memoize(f):
        class memodict(dict):
            def __missing__(self, key):
                self[key] = ret = f(key)
                return ret
        return memodict().__getitem__

    @memoize
    def adj(coord):
        x, y = coord
        return [(_x, _y) for _x in (x-1, x, x+1) for _y in (y-1, y, y+1)
                if (_x,_y) != (x, y) and 0 <= _x < 100 and 0 <= _y < 100]

    def lights(broken=False):

        lights = {(x,y) for y, row in enumerate(light_rows)
                  for x, c in enumerate(row) if c == '#'}
        corners = {(0,0), (0,99), (99,0), (99,99)}
        if broken: lights = corners | lights

        def neighbours(x, y):
            return sum((_x,_y) in lights for _x,_y in adj((x,y)))

        for _ in range(100):
            lights = {(x,y) for x in range(100) for y in range(100)
                     if (x,y) in lights and neighbours(x,y) in (2, 3)
                     or (x,y) not in lights and neighbours(x,y) == 3}
            if broken: lights = corners | lights

        return len(lights)


    printer.row(lights())
    printer.row(lights(broken=True))


#===============================================================================
# DAY 19

def day_19(s):

    reps, molecule = s.strip().split('\n\n')
    reps = re.findall(r'(\w+) => (\w+)', s)
    atoms = re.findall(r'([A-Z][a-z]*)', molecule)

    S = set()
    for element, rep in reps:
        for i in (i for i, atom in enumerate(atoms) if atom == element):
            y = atoms[:]
            y[i] = rep
            S.add(''.join(y))

    printer.row(len(S))

    '''
    PART 2 - replacement rules
    a => b c                    +1
    a => b Rn c Ar              +1 (+Rn +Ar)
    a => b Rn c Y d Ar          +1 (+Rn +Ar) (+1Y +1)
    a => b Rn c Y d Y e Ar      +1 (+Rn +Ar) (+2Y +2)
    '''
    n_reps = len(atoms) - atoms.count('Rn') - atoms.count('Ar') - 2*atoms.count('Y') - 1
    printer.row(n_reps)


#===============================================================================
# DAY 20

def day_20(s):

    presents = int(s)

    def robins_inequality(n):
        return exp(0.57721566490153286060651209008240243104215933593992)*n*log(log(n))

    def house(lo, hi, target, *, part=1):

        houses = [i for i in range(lo, hi, 1)]
        end    = len(houses)
        visits = [0]*end

        for i in range(houses[-1], 1, -1):
            start = i*ceil(houses[0]/i)-houses[0]
            if part == 2:
                end = min(end, 50*i - houses[0])
            for j in range(start, end, i):
                visits[j] += i

        return next((houses[0]+i for i, v in enumerate(visits) if v >= target), None)

    def bsearch(lo, hi):
        while lo < hi:
            mid = (lo+hi)//2
            midval = robins_inequality(mid)
            if midval < target:
                lo = mid+1
            elif midval > target:
                hi = mid
        return lo

    target = presents // 10
    lo = bsearch(5040, target)
    hi = int(lo*1.1)
    house_1 = house(lo, hi, target)
    printer.row(house_1)

    target = presents // 11
    lo = house_1
    hi = int(lo*1.1)
    house_2 = house(lo, hi, target, part=2)
    printer.row(house_2)


#===============================================================================
# DAY 21

def day_21(s):

    boss_hp, boss_dmg, boss_armor = map(int, re.match(r'Hit Points: (\d+)\W*Damage: (\d+)\W*Armor: (\d+)', s).groups())
    player_hp    = 100
    equipment    = {}
    boss_total   = boss_hp*boss_dmg + player_hp*boss_armor

    def item(cost, attack, defence):
        return {
            'cost' : cost,
            'att'  : attack,
            'def'  : defence,
            'bonus': player_hp*attack + boss_hp*defence
            }

    weapons = [
        item(  8, 4, 0),
        item( 10, 5, 0),
        item( 25, 6, 0),
        item( 40, 7, 0),
        item( 74, 8, 0)
    ]
    armours = [
        item(  0, 0, 0),
        item( 13, 0, 1),
        item( 31, 0, 2),
        item( 53, 0, 3),
        item( 75, 0, 4),
        item(102, 0, 5)
    ]
    rings = [
        item(  0, 0, 0),
        item( 25, 1, 0),
        item( 50, 2, 0),
        item(100, 3, 0),
        item( 20, 0, 1),
        item( 40, 0, 2),
        item( 80, 0, 3)
    ]


    def equipment_total(stat):
        weapon = weapons[equipment['weapon']]
        armour = armours[equipment['armour']]
        ring_1 = rings[equipment['ring_1']]
        ring_2 = rings[equipment['ring_2']]
        return weapon[stat] + armour[stat] + ring_1[stat] + ring_2[stat]


    # Part 1 : win boss (equipment_total('bonus') >= boss_total) with minimum total cost

    equipment['weapon'] = 0
    equipment['armour'] = 0
    equipment['ring_1'] = 0
    equipment['ring_2'] = 0

    def best_upgrade(items, slot, upgrade):

        equipped = items[equipment[slot]]
        for i, item in enumerate(items):
            diff = item['bonus'] - equipped['bonus']
            if diff > 0:
                ratio = diff / (item['cost'] - equipped['cost'])
                if ratio > upgrade['ratio']:
                    upgrade['slot']  = slot
                    upgrade['idx']   = i
                    upgrade['ratio'] = ratio

    while equipment_total('bonus') < boss_total:

        upgrade = { 'slot':'', 'idx':-1, 'ratio':-1 }
        best_upgrade(weapons, 'weapon', upgrade)
        best_upgrade(armours, 'armour', upgrade)
        best_upgrade(rings,   'ring_1', upgrade)
        best_upgrade(rings,   'ring_2', upgrade)
        equipment[upgrade['slot']] = upgrade['idx']

    printer.row(equipment_total('cost'))


    # Part 2 : lose to boss (equipment_total('bonus') < boss_total) with maximum total cost

    equipment['weapon'] = 0
    equipment['armour'] = 0
    equipment['ring_1'] = 0
    equipment['ring_2'] = 0

    def worst_upgrade(items, slot, upgrade):

        equipped = items[equipment[slot]]
        for i, item in enumerate(items):
            diff = item['bonus'] - equipped['bonus']
            if diff > 0 and equipment_total('bonus') + diff < boss_total:
                ratio = diff / (item['cost'] - equipped['cost'])
                if ratio < upgrade['ratio']:
                    upgrade['slot']  = slot
                    upgrade['idx']   = i
                    upgrade['ratio'] = ratio

    while True:

        upgrade = { 'slot':'', 'idx':-1, 'ratio':float('inf') }
        worst_upgrade(weapons, 'weapon', upgrade)
        worst_upgrade(armours, 'armour', upgrade)
        worst_upgrade(rings,   'ring_1', upgrade)
        worst_upgrade(rings,   'ring_2', upgrade)
        if upgrade['idx'] < 0: break
        equipment[upgrade['slot']] = upgrade['idx']

    printer.row(equipment_total('cost'))


#===============================================================================
# DAY 22

def day_22(s):

    bhp, bdmg = map(int, re.match(r'Hit Points: (\d+)\W*Damage: (\d+)', s).groups())

    def sim(actions, part):
        boss_hp, boss_dmg = bhp, bdmg
        hp, mana, armor = 50, 500, 0
        turn, turn_c = 0, 0
        mana_spent = 0
        poison_left, shield_left, recharge_left = 0, 0, 0
        my_turn = True
        spell_cost = {'M': 53, 'D': 73, 'S': 113, 'P': 173, 'R': 229}

        while True:
            if len(actions)-1 < turn_c:
                print('out of moves')
                return 0
            if poison_left:
                poison_left = max(poison_left - 1, 0)
                boss_hp -= 3
            if shield_left:
                shield_left = max(shield_left - 1, 0)
                armor = 7
            else:
                armor = 0
            if recharge_left:
                recharge_left = max(recharge_left - 1, 0)
                mana += 101
            if my_turn:
                if part == 2:
                    hp -= 1
                    if hp <= 0:
                        return 0
                action = actions[turn_c]
                mana -= spell_cost[action]
                mana_spent += spell_cost[action]
                if action == 'M':
                    boss_hp -= 4
                elif action == 'D':
                    boss_hp -= 2
                    hp += 2
                elif action == 'S':
                    if shield_left:
                        return 0
                    shield_left = 6
                elif action == 'P':
                    if poison_left:
                        return 0
                    poison_left = 6
                elif action == 'R':
                    if recharge_left:
                        return 0
                    recharge_left = 5
                if mana < 0:
                    return 0
            if boss_hp <= 0:
                return mana_spent
            if not my_turn:
                hp -= max(boss_dmg - armor, 1)
                if hp <= 0:
                    return 0
            if my_turn:
                turn_c += 1
            my_turn ^= True
            turn += 1

    def iterate_actions(pos):
        actions[pos] = 'DSPRM'['MDSPR'.index(actions[pos])]
        if actions[pos] == 'M':
            if pos+1 <= len(actions):
                iterate_actions(pos+1)

    for part in (1, 2):
        actions = ['M'] * 20
        min_spent = float('inf')
        for i in range(1000000):
            result = sim(actions, part)
            if result:
                min_spent = min(result, min_spent)
            iterate_actions(0)
        printer.row(min_spent)


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
solver("input/15.txt", day_15)
solver("input/16.txt", day_16)
solver("input/17.txt", day_17)
solver("input/18.txt", day_18)
solver("input/19.txt", day_19)
solver("input/20.txt", day_20)
solver("input/21.txt", day_21)
solver("input/22.txt", day_22)
