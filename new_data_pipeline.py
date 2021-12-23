import numpy as np
from pathlib import Path
from random import shuffle
from tqdm import tqdm, trange
import deflate
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

RECORD_SIZE = 8356
ARRAY_SHAPES_WITHOUT_BATCH = [(112, 64), (1858,), (3,), (3,), (1,)]


def file_generator(file_list, random):
    while True:
        if random:
            shuffle(file_list)
        else:
            file_list = sorted(file_list)
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

    z = np.concatenate((z_q_w, z_d, z_q_l), axis=1)

    # Outcome distribution needs to be calculated from q and d.
    best_q = np.ascontiguousarray(raw[:, 8284: 8284 + 4]).view(dtype=np.float32)
    best_d = np.ascontiguousarray(raw[:, 8292: 8292 + 4]).view(dtype=np.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = np.concatenate((best_q_w, best_d, best_q_l), axis=1)

    ply_count = np.ascontiguousarray(raw[:, 8304: 8304 + 4]).view(dtype=np.float32)
    return z, q, ply_count


def extract_inputs_outputs_if1(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 are castling + stm, all of which simply copy the byte value to all squares.
    unit_planes = extract_byte_planes(raw).astype(np.float32)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    inputs = np.concatenate([bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1).reshape([-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return inputs, policy, z, q, ply_count


def offset_generator(batch_size, record_size, skip_factor, random):
    initial_offset = 0
    while True:
        if random:
            retained_indices = np.random.choice(batch_size * skip_factor, size=batch_size, replace=False)
        else:
            retained_indices = np.array([i * skip_factor for i in range(batch_size)])
        retained_indices = np.sort(retained_indices)
        next_offset = batch_size * skip_factor - retained_indices[-1]  # Bump us up to the end of the current skip-batch
        skip_offsets = np.diff(retained_indices, prepend=0)
        skip_offsets[0] += initial_offset
        for offset in skip_offsets:
            yield offset * record_size
        initial_offset = next_offset


def data_generator(files, batch_size, skip_factor, validation):
    # This is a singlethreaded generator for debugging
    file_gen = file_generator(files, random=not validation)
    offset_gen = offset_generator(batch_size=batch_size, record_size=RECORD_SIZE, skip_factor=skip_factor,
                                  random=not validation)
    data = np.zeros((batch_size, RECORD_SIZE), dtype=np.uint8)
    current_file = next(file_gen)
    file_ptr = 0
    data_ptr = 0
    offset = next(offset_gen)
    while True:
        if offset + file_ptr < len(current_file):
            data[data_ptr] = np.frombuffer(current_file[offset + file_ptr: offset + file_ptr + RECORD_SIZE],
                                           dtype=np.uint8)
            data_ptr += 1
            if data_ptr == batch_size:
                yield extract_inputs_outputs_if1(data)
                data.fill(0)
                data_ptr = 0
            file_ptr += offset + RECORD_SIZE
            offset = next(offset_gen)
        else:
            offset -= len(current_file) - file_ptr
            current_file = next(file_gen)
            file_ptr = 0


def data_worker(files, batch_size, skip_factor, array_ready_event, main_process_access_event, shared_array_names, validation):
    shared_mem = [SharedMemory(name=name, create=False) for name in shared_array_names]
    array_shapes = [[batch_size] + list(shape) for shape in ARRAY_SHAPES_WITHOUT_BATCH]
    shared_arrays = [np.ndarray(shape, dtype=np.float32, buffer=mem.buf)
                     for shape, mem in zip(array_shapes, shared_mem)]
    file_gen = file_generator(files, random=not validation)
    offset_gen = offset_generator(batch_size=batch_size, record_size=RECORD_SIZE, skip_factor=skip_factor, random=not validation)
    data = np.zeros((batch_size, RECORD_SIZE), dtype=np.uint8)
    current_file = next(file_gen)
    file_ptr = 0
    data_ptr = 0
    offset = next(offset_gen)
    while True:
        if offset + file_ptr < len(current_file):
            data[data_ptr] = np.frombuffer(current_file[offset + file_ptr: offset + file_ptr + RECORD_SIZE],
                                           dtype=np.uint8)
            data_ptr += 1
            if data_ptr == batch_size:
                processed_batch = extract_inputs_outputs_if1(data)
                main_process_access_event.wait()
                main_process_access_event.clear()
                for batch_array, shared_array in zip(processed_batch, shared_arrays):
                    shared_array[:] = batch_array
                    array_ready_event.set()
                data.fill(0)
                data_ptr = 0
            file_ptr += offset + RECORD_SIZE
            offset = next(offset_gen)
        else:
            offset -= len(current_file) - file_ptr
            current_file = next(file_gen)
            file_ptr = 0


def multiprocess_generator(chunk_dir, batch_size, num_workers, skip_factor, shuffle_buffer_size, validation=False):
    assert shuffle_buffer_size % batch_size == 0  # This simplifies my life later on
    print("Scanning directory for game data chunks...")
    files = list(tqdm(chunk_dir.glob('**/*.gz'), desc="Files found", unit=" files"))
    print("Done!")
    if validation:
        files = sorted(files)
    else:
        shuffle(files)
    worker_file_lists = [files[i::num_workers] for i in range(num_workers)]
    ctx = get_context('spawn')  # For Windows compatibility
    array_ready_events = []
    main_process_access_events = []
    shared_arrays = []
    shared_mem = []
    array_shapes = [[batch_size] + list(shape) for shape in ARRAY_SHAPES_WITHOUT_BATCH]
    array_sizes = [int(np.prod(shape)) * 4 for shape in array_shapes]
    shuffle_buffer_shapes = [[shuffle_buffer_size] + list(shape[1:]) for shape in array_shapes]
    shuffle_buffers = [np.zeros(shape=shape, dtype=np.float32) for shape in shuffle_buffer_shapes]

    for i in trange(num_workers, desc="Initializing worker processes"):
        array_ready_event = ctx.Event()
        main_process_access_event = ctx.Event()
        main_process_access_event.set()
        array_ready_events.append(array_ready_event)
        main_process_access_events.append(main_process_access_event)
        process_shared_mem = [SharedMemory(size=size, create=True) for size in array_sizes]
        process_shared_arrays = [np.ndarray(array_shapes[i], dtype=np.float32, buffer=process_shared_mem[i].buf)
                                 for i in range(len(array_shapes))]
        shared_mem.append(process_shared_mem)
        shared_arrays.append(process_shared_arrays)
        shared_mem_names = [mem.name for mem in process_shared_mem]
        process = ctx.Process(target=data_worker, kwargs={
            "files": worker_file_lists[i], "skip_factor": skip_factor,
            "batch_size": batch_size, "array_ready_event": array_ready_event,
            "main_process_access_event": main_process_access_event,
            "shared_array_names": shared_mem_names,
            "validation": validation}, daemon=True)
        process.start()

    for i in trange(shuffle_buffer_size // batch_size, desc="Filling shuffle buffer"):
        proc = i % num_workers
        array_ready_events[proc].wait()
        for array, shuffle_buffer in zip(shared_arrays[proc], shuffle_buffers):
            shuffle_buffer[i * batch_size: (i + 1) * batch_size] = array
        array_ready_events[proc].clear()
        main_process_access_events[proc].set()

    while True:
        for array_ready_event, main_process_access_event, shared_arrs in zip(array_ready_events, main_process_access_events, shared_arrays):
            if not array_ready_event.is_set():
                continue
            random_indices = np.random.choice(shuffle_buffer_size, size=batch_size, replace=False)
            batch = tuple([shuffle_buffer[random_indices] for shuffle_buffer in shuffle_buffers])
            yield batch
            for arr, shuffle_buffer in zip(shared_arrs, shuffle_buffers):
                shuffle_buffer[random_indices] = arr
            array_ready_event.clear()
            main_process_access_event.set()


def make_callable(chunk_dir, batch_size, num_workers, skip_factor, shuffle_buffer_size):
    # Because tf.data needs to be able to reinitialize
    def return_gen():
        return multiprocess_generator(chunk_dir=chunk_dir, batch_size=batch_size,
                                      num_workers=num_workers, skip_factor=skip_factor,
                                      shuffle_buffer_size=shuffle_buffer_size)
    return return_gen


def main():
    test_dir = Path("/home/matt/leela_training_data/rescored/training-run1-test60-20210701-0017")
    batch_size = 1024
    num_workers = 4
    shuffle_buffer_size = 2 ** 17
    skip_factor = 32
    gen = multiprocess_generator(chunk_dir=test_dir, batch_size=batch_size,
                                 num_workers=num_workers, skip_factor=skip_factor,
                                 shuffle_buffer_size=shuffle_buffer_size)
    for _ in tqdm(gen):
        pass


if __name__ == '__main__':
    main()
