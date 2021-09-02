import numpy as np
import gzip
from pathlib import Path
from random import shuffle
from tqdm import tqdm
import deflate

RECORD_SIZE = 8356
SKIP_FACTOR = 32


def file_generator(file_list):
    while True:
        shuffle(file_list)
        for file in file_list:
            # yield gzip.open(file, 'rb').read()
            yield deflate.gzip_decompress(file.read_bytes())


def extract_rule50_zero_one(raw):
    # Tested equivalent but there were a lot of zeros so I'm unsure
    # rule50 count plane.
    rule50_plane = raw[:, 8277: 8277 + 1].reshape(-1, 1, 1, 1).astype(np.float32) / 99.
    rule50_plane = np.tile(rule50_plane, [1, 1, 8, 8])
    # zero plane and one plane
    zero_plane = np.zeros_like(rule50_plane)
    one_plane = np.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_byte_planes(raw):
    # Checked and confirmed equivalent to the existing extract_byte_planes
    # 5 bytes in input are expanded and tiled
    planes = raw[:, 8272: 8272 + 5].reshape(-1, 5, 1, 1)
    unit_planes = np.tile(planes, [1, 1, 8, 8])
    return unit_planes


def extract_policy_bits(raw):
    # Checked and confirmed equivalent to the existing extract_policy_bits
    # Next 7432 are easy, policy extraction.
    policy = np.ascontiguousarray(raw[:, 8: 8 + 7432]).view(dtype=np.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = raw[:, 7440: 7440+832].reshape((-1, 104, 8))
    bit_planes = np.unpackbits(bit_planes, axis=-1).reshape((-1, 104, 8, 8))
    return policy, bit_planes


def extract_outputs(raw):
    # Checked and confirmed equivalent to the existing extract_outputs
    # Result distribution needs to be calculated from q and d.
    z_q = np.ascontiguousarray(raw[:, 8308: 8308 + 4]).view(dtype=np.float32)
    z_d = np.ascontiguousarray(raw[:, 8312: 8312 + 4]).view(dtype=np.float32)
    z_q_w = 0.5 * (1.0 - z_d + z_q)
    z_q_l = 0.5 * (1.0 - z_d - z_q)

    z = np.concatenate([z_q_w, z_d, z_q_l], axis=1)

    # Outcome distribution needs to be calculated from q and d.
    best_q = np.ascontiguousarray(raw[:, 8284: 8284 + 4]).view(dtype=np.float32)
    best_d = np.ascontiguousarray(raw[:, 8292: 8292 + 4]).view(dtype=np.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = np.concatenate([best_q_w, best_d, best_q_l], axis=1)

    ply_count = np.ascontiguousarray(raw[:, 8304: 8304 + 4]).view(dtype=np.float32)
    return z, q, ply_count


def extract_inputs_outputs_if1(raw_list):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    raw = np.stack([np.frombuffer(arr, dtype=np.uint8) for arr in raw_list], axis=0)

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 are castling + stm, all of which simply copy the byte value to all squares.
    unit_planes = extract_byte_planes(raw).astype(np.float32)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    inputs = np.concatenate([bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1).reshape([-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return inputs, policy, z, q, ply_count


def offset_generator(batch_size, record_size):
    initial_offset = 0
    while True:
        retained_indices = np.random.choice(batch_size * SKIP_FACTOR, size=batch_size, replace=False)
        retained_indices = np.sort(retained_indices)
        next_offset = batch_size * SKIP_FACTOR - retained_indices[-1]  # Bump us up to the end of the current skip-batch
        skip_offsets = np.diff(retained_indices, prepend=0)
        skip_offsets[0] += initial_offset
        for offset in skip_offsets:
            yield offset * record_size
        initial_offset = next_offset


def data_generator(files, batch_size):
    file_gen = file_generator(files)
    offset_gen = offset_generator(batch_size=1024, record_size=RECORD_SIZE)
    data = []
    current_file = next(file_gen)
    file_ptr = 0
    offset = next(offset_gen)
    while True:
        if offset + file_ptr < len(current_file):
            data.append(current_file[offset + file_ptr: offset + file_ptr + RECORD_SIZE])
            if len(data) == batch_size:
                yield extract_inputs_outputs_if1(data)
                data = []
            file_ptr += offset + RECORD_SIZE
            offset = next(offset_gen)
        else:
            offset -= len(current_file) - file_ptr
            current_file = next(file_gen)
            file_ptr = 0


def main():
    test_dir = Path("/home/matt/leela_training_data/rescored/training-run1-test60-20210701-0017/")
    files = list(test_dir.glob('**/*.gz'))
    data_gen = data_generator(files, batch_size=1024)
    for batch in tqdm(data_gen):
        pass


if __name__ == '__main__':
    main()