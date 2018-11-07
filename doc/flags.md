Here we'll describe in detail the full set of command line flags available for preprocessing, training, and sampling.

# Preprocessing
The preprocessing script `scripts/preprocess.py` accepts the following command-line flags:

- `--input_txt`: Path to the text file to be used for training. Default is the `tiny-shakespeare.txt` dataset.
- `--output_h5`: Path to the HDF5 file where preprocessed data should be written.
- `--output_json`: Path to the JSON file where preprocessed data should be written.
- `--val_frac`: What fraction of the data to use as a validation set; default is `0.1`.
- `--test_frac`: What fraction of the data to use as a test set; default is `0.1`.
- `--quiet`: If you pass this flag then no output will be printed to the console.
- `--input_json`: Path to the JSON file from a previous run to reuse vocabulary
- `--freeze-vocab`: Don't allow expanding vocabulary.
- `--encoding`: Specify input encoding, `utf-8` or `bytes`


# Training
The training script `train.lua` accepts the following command-line flags:

**Data options**:
- `-input_h5`, `-input_json`: Paths to the HDF5 and JSON files output from the preprocessing script.
- `-batch_size`: Number of sequences to use in a minibatch; default is 50.
- `-seq_length`: Number of timesteps for which the recurrent network is unrolled for backpropagation through time.
- `-seq_offset`: Shift data by `seq_length/2` every other epoch
- `-shuffle_data`: Order batches randomly every epoch

**Model options**:

- `-init_from`: Path to a checkpoint file from a previous run of `train.lua`. Use this to continue training from an existing checkpoint; if this flag is passed then the other flags in this section will be ignored and the architecture from the existing checkpoint will be used instead.
- `-reset_iterations`: Set this to 0 to restore the iteration counter of a previous run. Default is 1 (do not restore iteration counter). Only applicable if `-init_from` option is used.
- `-model_type`: The type of recurrent network to use; either `lstm` (default) or `rnn`. `lstm` is slower but better. `gridgru` is also supported.
- `-wordvec_size`: Dimension of learned word vector embeddings; default is 64. You probably won't need to change this.
- `-rnn_size`: The number of hidden units in the RNN; default is 128. Larger values (256 or 512) are commonly used to learn more powerful models and for bigger datasets, but this will significantly slow down computation.
- `-dropout`: Amount of dropout regularization to apply after each RNN layer; must be in the range `0 <= dropout < 1`. Setting `dropout` to 0 disables dropout, and higher numbers give a stronger regularizing effect.
- `-num_layers`: The number of layers present in the RNN; default is 2.
- `-zoneout`: Specify zoneout (GRIDGRU only, 0.01 seems to work well)

**Optimization options**:

- `-max_epochs`: How many training epochs to use for optimization. Default is 50.
- `-learning_rate`: Learning rate for optimization. Default is `2e-3`.
- `-grad_clip`: Maximum value for gradients; default is 5. Set to 0 to disable gradient clipping.
- `-lr_decay_every`: How often to decay the learning rate, in epochs; default is 5.
- `-lr_decay_factor`: How much to decay the learning rate. After every `lr_decay_every` epochs, the learning rate will be multiplied by the `lr_decay_factor`; default is 0.5.

**Output options**:

- `-print_every`: How often to print status message, in iterations. Default is 1.
- `-checkpoint_name`: Base filename for saving checkpoints; default is `cv/checkpoint`. This will create checkpoints named - `cv/checkpoint_1000.t7`, `cv/checkpoint_1000.json`, etc.
- `-checkpoint_every`: How often to save intermediate checkpoints. Default is 1000; set to 0 to disable intermediate checkpointing. Note that we always save a checkpoint on the final iteration of training.

**Benchmark options**:

- `-speed_benchmark`: Set this to 1 to test the speed of the model at every iteration. This is disabled by default because it requires synchronizing the GPU at every iteration, which incurs a performance overhead. Speed benchmarking results will be printed and also stored in saved checkpoints.
- `-memory_benchmark`: Set this to 1 to test the GPU memory usage at every iteration. This is disabled by default because like speed benchmarking it requires GPU synchronization. Memory benchmarking results will be printed and also stored in saved checkpoints. Only available when running in GPU mode.

**Backend options**:

- `-gpu`: The ID of the GPU to use (zero-indexed). Default is 0. Set this to -1 to run in CPU-only mode
- `-gpu_backend`: The GPU backend to use; either `cuda` or `opencl`. Default is `cuda`.
- `-gpu_opt`: GPU to use for running the optimizer to save memory on the primary GPU. Set to -1 to run on CPU or -2 (default) to run on the same as `-gpu`
- `-swaprnn`: Reuse temporary GPU tensors between layers while swapping them out to host memory. GRIDGRU only.
- `-low_mem_dropout`: Use a slightly slower but GPU memory-saving dropout implementation.

# Sampling
The sampling script `sample.lua` accepts the following command-line flags:

- `-checkpoint`: Path to a `.t7` checkpoint file from `train.lua`
- `-length`: The length of the generated text, in characters.
- `-start_text`: You can optionally start off the generation process with a string; if this is provided the start text will be processed by the trained network before we start sampling. Without this flag, the first character is chosen randomly.
- `-sample`: Set this to 1 to sample from the next-character distribution at each timestep; set to 0 to instead just pick the argmax at every timestep. Sampling tends to produce more interesting results.
- `-temperature`: Softmax temperature to use when sampling; default is 1. Higher temperatures give noiser samples. Not used when using argmax sampling (`sample` set to 0).
- `-gpu`: The ID of the GPU to use (zero-indexed). Default is 0. Set this to -1 to run in CPU-only mode.
- `-gpu_backend`: The GPU backend to use; either `cuda` or `opencl`. Default is `cuda`.
- `-verbose`: By default just the sampled text is printed to the console. Set this to 1 to also print some diagnostic information.
- `-hide_start_text`: Don't print `start_text` on the output
- `-read_start_text`: Read `start_text` value from stdin before generating
- `-stop_on_newline`: Stop generating output once hitting a newline
- `-bytes`: Set to 1 when using data preprocessed with `--encoding bytes`

# Server
The server script `server.lua` can be used as a chat bot backend from inetd, or directly used as an interactive application. By default, a line of text is generated when receiving an empty line.

- `-checkpoint`, `-sample`, `-temperature`, `bytes`: same as `sample.lua`
- `-verbose`: Print extra information. Values of 0 to 3 are supported when using `multi_count`
- `-maxlength`: Maximum length of generated lines
- `-interactive`: Read input using readline
- `-autoreply`: Generate a line after every line of input
- `-start_text`: Feed a line of text into the model after startup
- `-color`: Color output characters based on their entropy
- `-multi_count`: Generate N output lines in parallel, then pick the best line
- `-benchmark`: Print time taken to generate each character
- `-relevance_sampling`: Penalize generic responses when sampling characters
- `-relevance_selection`: Penalize generic responses when selecting output line when using `multi_count`
- `-gpu`: Set to 0 to use CPU, 1 to use GPU.
- `-commands`: Interpret `/!save`, `/!load` and `/!reset` commands on input
- `-savedir`: Directory for model state files when using `commands`
- `-ksm`: Mark weight tensors as shareable on Linux, to allow Kernel Samepage Merging to merge the pages between multiple instances
- `-lineprefix`: Start every output line with a prefix

# Multisample
The script `multisample.lua` allows generating multiple output files in parallel.

- `-checkpoint`, `-sample`, `-temperature`, `-bytes`, `-start_text`, `-length`: Same as `sample.lua`
- `-count`: Number of output files to generate
- `-print_every`: Print status every N characters
- `-output_file`: Set output file. `#` is replaced by the file number.
- `-verbose`: Show lines as they are generated
- `-gpu`: Set to 1 to run on GPU

# Utilities
- `strip_gradients.lua`: Strip gradient tensors from a model. `-i` and `-o` specify input and output files.
- `build_ensemble.lua`: Builds an ensemble of multiple models. `-output` specifies the output file, input files are read from stdin.


