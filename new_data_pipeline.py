import numpy as np
import gzip
from pathlib import Path
from random import shuffle
from tqdm import tqdm, trange
import deflate
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

RECORD_SIZE = 8356
SKIP_FACTOR = 32
NUM_WORKERS = 12
BATCH_SIZE = 1024
SHUFFLE_BUFFER_SIZE = 2 ** 19
assert SHUFFLE_BUFFER_SIZE % BATCH_SIZE == 0  # This simplifies my life later on
ARRAY_SHAPES = [(BATCH_SIZE, 112, 64), (BATCH_SIZE, 1858), (BATCH_SIZE, 3), (BATCH_SIZE, 3), (BATCH_SIZE, 1)]
ARRAY_SIZES = [int(np.prod(shape)) * 4 for shape in ARRAY_SHAPES]


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
    offset_gen = offset_generator(batch_size=BATCH_SIZE, record_size=RECORD_SIZE)
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


def data_worker(files, batch_size, array_ready_event, main_process_access_event, shared_array_names):
    shared_mem = [SharedMemory(name=name, create=False) for name in shared_array_names]
    shared_arrays = [np.ndarray(shape, dtype=np.float32, buffer=mem.buf)
                     for shape, mem in zip(ARRAY_SHAPES, shared_mem)]
    file_gen = file_generator(files)
    offset_gen = offset_generator(batch_size=BATCH_SIZE, record_size=RECORD_SIZE)
    data = []
    current_file = next(file_gen)
    file_ptr = 0
    offset = next(offset_gen)
    while True:
        if offset + file_ptr < len(current_file):
            data.append(current_file[offset + file_ptr: offset + file_ptr + RECORD_SIZE])
            if len(data) == batch_size:
                processed_batch = extract_inputs_outputs_if1(data)
                main_process_access_event.wait()
                main_process_access_event.clear()
                for batch_array, shared_array in zip(processed_batch, shared_arrays):
                    shared_array[:] = batch_array
                    array_ready_event.set()
                data = []
            file_ptr += offset + RECORD_SIZE
            offset = next(offset_gen)
        else:
            offset -= len(current_file) - file_ptr
            current_file = next(file_gen)
            file_ptr = 0


def multiprocess_generator(chunk_dir):
    print("Scanning directory for game data chunks...")
    files = list(tqdm(chunk_dir.glob('**/*.gz'), desc="Files found", unit=" files"))
    print("Done!")
    shuffle(files)
    worker_file_lists = [files[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    ctx = get_context('spawn')  # For Windows compatibility
    array_ready_events = []
    main_process_access_events = []
    shared_arrays = []
    shared_mem = []
    shuffle_buffer_shapes = [[SHUFFLE_BUFFER_SIZE] + list(shape[1:]) for shape in ARRAY_SHAPES]
    shuffle_buffers = [np.zeros(shape=shape, dtype=np.float32) for shape in shuffle_buffer_shapes]

    for i in range(NUM_WORKERS):
        array_ready_event = ctx.Event()
        main_process_access_event = ctx.Event()
        main_process_access_event.set()
        array_ready_events.append(array_ready_event)
        main_process_access_events.append(main_process_access_event)
        process_shared_mem = [SharedMemory(size=size, create=True) for size in ARRAY_SIZES]
        process_shared_arrays = [np.ndarray(ARRAY_SHAPES[i], dtype=np.float32, buffer=process_shared_mem[i].buf)
                                 for i in range(len(ARRAY_SHAPES))]
        shared_mem.append(process_shared_mem)
        shared_arrays.append(process_shared_arrays)
        shared_mem_names = [mem.name for mem in process_shared_mem]
        process = ctx.Process(target=data_worker, kwargs={
            "files": worker_file_lists[i],
            "batch_size": BATCH_SIZE, "array_ready_event": array_ready_event,
            "main_process_access_event": main_process_access_event,
            "shared_array_names": shared_mem_names}, daemon=True)
        process.start()

    for i in trange(SHUFFLE_BUFFER_SIZE // BATCH_SIZE, desc="Filling shuffle buffer"):
        proc = i % NUM_WORKERS
        array_ready_events[proc].wait()
        for array, shuffle_buffer in zip(shared_arrays[proc], shuffle_buffers):
            shuffle_buffer[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = array
        array_ready_events[proc].clear()
        main_process_access_events[proc].set()

    while True:
        for array_ready_event, main_process_access_event, shared_arrs in zip(array_ready_events, main_process_access_events, shared_arrays):
            if not array_ready_event.is_set():
                continue
            random_indices = np.random.choice(SHUFFLE_BUFFER_SIZE, size=BATCH_SIZE, replace=False)
            batch = tuple([np.copy(shuffle_buffer[random_indices]) for shuffle_buffer in shuffle_buffers])
            for arr, shuffle_buffer in zip(shared_arrs, shuffle_buffers):
                shuffle_buffer[random_indices] = arr
            array_ready_event.clear()
            main_process_access_event.set()
            yield batch


def make_callable(chunk_dir):
    # Because tf.data needs to be able to reinitialize
    def return_gen():
        return multiprocess_generator(chunk_dir)
    return return_gen


def main():
    test_dir = Path("/home/matt/leela_training_data/rescored/training-run1-test60-20210701-0017")
    gen = multiprocess_generator(test_dir)
    for _ in tqdm(gen):
        pass


if __name__ == '__main__':
    main()
