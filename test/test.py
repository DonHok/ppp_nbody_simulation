import sys

expected = open(sys.argv[1], 'r')
result = open(sys.argv[2], 'r')

expected_lines = map(lambda x: x.strip(), expected.readlines())
result_lines = map(lambda x: x.strip(), result.readlines())

assert len(set(result_lines)) == len(result_lines)

for line in result_lines:
    assert line in expected_lines

