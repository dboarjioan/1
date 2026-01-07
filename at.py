import argparse
import numpy as np
import torch
import sys


def parse_ply_header_bin(f):
    f.seek(0)
    header_lines = []
    elements = {}
    current_elem = None

    while True:
        line = f.readline()
        if not line:
            raise EOFError("EOF before end_header")

        header_lines.append(line.decode('ascii', errors='ignore').rstrip("\r\n"))
        toks = header_lines[-1].split()

        if not toks:
            continue

        if toks[0] == "element":
            name = toks[1]
            count = int(toks[2])
            current_elem = name
            elements[current_elem] = {"count": count, "properties": []}

        elif toks[0] == "property":
            if toks[1] == "list":
                raise NotImplementedError("List properties not supported")
            elements[current_elem]["properties"].append((toks[1], toks[2]))

        elif toks[0] == "end_header":
            return header_lines, f.tell(), elements


def ply_type_to_dtype(t):
    mapping = {
        "char": np.int8, "uchar": np.uint8,
        "short": np.int16, "ushort": np.uint16,
        "int": np.int32, "uint": np.uint32,
        "float": np.float32, "double": np.float64
    }
    return mapping.get(t, None)


# -------------------------------------------------
# Read vertex block
# -------------------------------------------------
def read_vertex_block(path):
    with open(path, "rb") as f:
        header_lines, header_bytes, elements = parse_ply_header_bin(f)

        if "vertex" not in elements:
            raise ValueError("No vertex element in PLY")

        vertex_info = elements["vertex"]
        n_vertices = vertex_info["count"]
        props = vertex_info["properties"]

        np_dtypes = []
        for p in props:
            dt = ply_type_to_dtype(p[0])
            if dt is None:
                raise ValueError(f"Unknown PLY type: {p[0]}")
            np_dtypes.append((p[1], dt))

        struct_dtype = np.dtype(np_dtypes).newbyteorder("<")
        f.seek(header_bytes)

        data = f.read(n_vertices * struct_dtype.itemsize)
        arr = np.frombuffer(data, dtype=struct_dtype, count=n_vertices)
        remainder = f.read()

        return header_lines, header_bytes, elements, arr, remainder


# -------------------------------------------------
# Write back modified PLY
# -------------------------------------------------
def write_ply(in_path, out_path, header_bytes, struct_arr, remainder):
    with open(in_path, "rb") as fin:
        header = fin.read(header_bytes)

    with open(out_path, "wb") as fout:
        fout.write(header)
        fout.write(struct_arr.tobytes())
        fout.write(remainder)


# -------------------------------------------------
# Utility
# -------------------------------------------------
def find_property_indices(all_names, target_names):
    idx = []
    for nm in target_names:
        if nm not in all_names:
            raise ValueError(f"Property '{nm}' not found in PLY")
        idx.append(all_names.index(nm))
    return idx


def structured_to_matrix(struct_arr):
    names = list(struct_arr.dtype.names)
    cols = [struct_arr[name].astype(np.float32).reshape(-1,1) for name in names]
    return np.hstack(cols), names


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    parser.add_argument("--props", nargs="+", required=True,
                        help="names of gaussian parameters to attack (e.g. f_dc_0 f_dc_1 f_dc_2)")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=50, help="PGD steps (1 = FGSM)")
    parser.add_argument("--target", nargs="+", type=float, default=None)
    parser.add_argument("--clip", nargs=2, type=float, default=(0.0,1.0))
    args = parser.parse_args()

    # Load PLY
    header_lines, header_bytes, elements, struct_arr, remainder = read_vertex_block(args.input)
    mat, prop_names = structured_to_matrix(struct_arr)

    print(f"Loaded {mat.shape[0]} vertices with {mat.shape[1]} properties.")
    print("Properties:", prop_names)

    idxs = find_property_indices(prop_names, args.props)
    print("Attacked properties:", args.props)
    print("Indices:", idxs)

    # Extract parameters to modify
    P = mat[:, idxs].astype(np.float32)
    x = torch.tensor(P, dtype=torch.float32, requires_grad=True)

    # Build target
    if args.target is None:
        target = torch.zeros_like(x)
    else:
        t = torch.tensor(args.target, dtype=torch.float32)
        if t.numel() != len(idxs):
            print("ERROR: --target length must equal number of attacked properties!")
            sys.exit(1)
        target = t.reshape(1,-1).repeat(x.shape[0],1)

    # PGD/FGSM loop
 # PGD/FGSM untargeted — zmieniamy kolory jak najmocniej
    orig_x = x.detach().clone()

    print(f"\n=== Starting PGD for {args.steps} steps ===\n")

    for step in range(args.steps):
        x.requires_grad_(True)

        # untargeted loss — zwiększamy odległość od oryginału
        loss = torch.nn.functional.mse_loss(x, orig_x)

        print(f"[PGD] step {step+1}/{args.steps}    loss={loss.item():.6f}")

        # reset grad
        if x.grad is not None:
            x.grad.zero_()

        loss.backward()

        with torch.no_grad():
            # update = zwiększ różnicę
            x = x + args.eps * x.grad.sign()

            # ograniczenia PGD względem oryginału
            delta = x - orig_x
            delta = torch.clamp(delta,
                                min=args.clip[0] - orig_x,
                                max=args.clip[1] - orig_x)

            x = orig_x + delta

        x = x.detach()

    adv = x.cpu().numpy()


    # Insert modified columns back
    mat[:, idxs] = adv

    # Rebuild structured output PLY array
    out_struct = np.empty(mat.shape[0], dtype=struct_arr.dtype)
    for i, name in enumerate(prop_names):
        out_struct[name] = mat[:, i].astype(struct_arr.dtype.fields[name][0])

    # Save
    write_ply(args.input, args.output, header_bytes, out_struct, remainder)

    print(f"\n✔ Written adversarial PLY → {args.output}")


if __name__ == "__main__":
    main()
