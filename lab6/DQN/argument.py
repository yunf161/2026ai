def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--lr", default=0.0012, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--n_step", default=3, type=int, help='multi-step return length')
    parser.add_argument("--grad_norm_clip", default=5, type=float)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--buffer_size", default=32000, type=int)
    parser.add_argument("--target_update_freq", default=50, type=int)
    parser.add_argument("--epsilon_start", default=1.0, type=float)
    parser.add_argument("--epsilon_end", default=0.02, type=float)
    parser.add_argument("--epsilon_decay_steps", default=5000, type=int)
    parser.add_argument("--epsilon_accelerate_step", default=2500, type=int)
    parser.add_argument("--train_every", default=1, type=int)
    parser.add_argument("--warmup_steps", default=200, type=int)
    parser.add_argument("--lr_decay_rate_slow", default=0.98, type=float)
    parser.add_argument("--lr_decay_rate_fast", default=0.9, type=float)
    parser.add_argument("--lr_accelerate_step", default=5000, type=int)
    parser.add_argument("--lr_decay_every", default=1000, type=int)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser
