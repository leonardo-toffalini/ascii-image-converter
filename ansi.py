def eval_control_sequence(seq):
    # only supports [<r>;<g>;<b>] as of now
    r, g, b = list(map(int, seq.split(";")))
    return f"\x1b[38;2;{r};{g};{b}m"

def parse(string):
    res = ""
    i = 0
    while i < len(string):
        if string[i] == "[":
            j = i + 1
            buf = ""
            while string[j] != "]":
                buf += string[j]
                j += 1
                if j >= len(string):
                    assert False, "Unterminated control sequence"
            i = j + 1
            res += eval_control_sequence(buf)
        else:
            res += string[i]
            i += 1
    return res

def rich_print(string, **kwargs):
    parsed = parse(string)
    print(parsed, **kwargs)

if __name__ == "__main__":
    s = "[40;177;249]@"
    print(s)
    parsed = parse(s)
    print(parsed)
    rich_print(s)

