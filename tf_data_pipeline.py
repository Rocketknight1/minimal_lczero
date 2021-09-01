import tensorflow as tf
import multiprocessing as mp
from pathlib import Path

experimental_reads = max(2, mp.cpu_count() - 2) // 2

SKIP = 32
SKIP_MULTIPLE = 1024
SHUFFLE_SIZE = 2 ** 17


def extract_policy_bits(raw):
    # Next 7432 are easy, policy extraction.
    policy = tf.io.decode_raw(tf.strings.substr(raw, 8, 7432), tf.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = tf.expand_dims(
        tf.reshape(
            tf.io.decode_raw(tf.strings.substr(raw, 7440, 832), tf.uint8),
            [-1, 104, 8]), -1)
    bit_planes = tf.bitwise.bitwise_and(tf.tile(bit_planes, [1, 1, 1, 8]),
                                        [128, 64, 32, 16, 8, 4, 2, 1])
    bit_planes = tf.minimum(1., tf.cast(bit_planes, tf.float32))
    return policy, bit_planes


def extract_byte_planes(raw):
    # 5 bytes in input are expanded and tiled
    unit_planes = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8272, 5), tf.uint8), -1),
        -1)
    unit_planes = tf.tile(unit_planes, [1, 1, 8, 8])
    return unit_planes


def extract_rule50_zero_one(raw):
    # rule50 count plane.
    rule50_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8277, 1), tf.uint8), -1),
        -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 99.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_outputs(raw):
    # Result distribution needs to be calculated from q and d.
    z_q = tf.io.decode_raw(tf.strings.substr(raw, 8308, 4), tf.float32)
    z_d = tf.io.decode_raw(tf.strings.substr(raw, 8312, 4), tf.float32)
    z_q_w = 0.5 * (1.0 - z_d + z_q)
    z_q_l = 0.5 * (1.0 - z_d - z_q)

    z = tf.concat([z_q_w, z_d, z_q_l], 1)

    # Outcome distribution needs to be calculated from q and d.
    best_q = tf.io.decode_raw(tf.strings.substr(raw, 8284, 4), tf.float32)
    best_d = tf.io.decode_raw(tf.strings.substr(raw, 8292, 4), tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    ply_count = tf.io.decode_raw(tf.strings.substr(raw, 8304, 4), tf.float32)
    return z, q, ply_count


def extract_inputs_outputs_if1(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 are castling + stm, all of which simply copy the byte value to all squares.
    unit_planes = tf.cast(extract_byte_planes(raw), tf.float32)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return inputs, policy, z, q, ply_count


def semi_sample(x):
    return tf.slice(tf.random.shuffle(x), [0], [SKIP_MULTIPLE])


def read(x):
    return tf.data.FixedLengthRecordDataset(
        x,
        8356,
        compression_type='GZIP',
        num_parallel_reads=experimental_reads)


def get_dataset(target_path, batch_size):
    target_path = Path(target_path)  # In case it's not already a Path object
    train_chunks = [str(pth) for pth in target_path.glob('**/*.gz')][:1_000_000]
    extractor = extract_inputs_outputs_if1
    #
    # Line 1: Shuffle the list of input files, repeat, and make batches of 256 files at a time
    # Line 2: Pass a list of 256 files to tf.data.FixedLengthRecordDataset. Interleave results from many of these calls.
    # Line 3: For each set of 32 * 1024 positions, subsample so we only take 1024 of them
    # Line 4  Shuffle with a large shuffle buffer
    # Line 5: Batch the output data and map each batch through the extractor
    train_dataset = tf.data.Dataset.from_tensor_slices(train_chunks).shuffle(len(train_chunks)).repeat().batch(256) \
        .interleave(read, num_parallel_calls=1) \
        .batch(SKIP_MULTIPLE * SKIP).map(semi_sample).unbatch() \
        .shuffle(SHUFFLE_SIZE) \
        .batch(batch_size).map(extractor)

    # Dataset returns input_planes, policy label, wdl label, q label, moves left label
    return train_dataset


if __name__ == '__main__':
    from tqdm import tqdm
    import sys
    from pathlib import Path
    target_dir = Path(sys.argv[1])
    assert target_dir.is_dir()
    print("Testing data pipeline...")
    print(f"Loading from directory {target_dir}")
    dataset = get_dataset(target_dir, batch_size=1024)
    for _ in tqdm(dataset):
        pass
