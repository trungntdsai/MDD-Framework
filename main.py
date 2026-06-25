import argparse

from trainer import MDDTrainer


def build_args():
    parser = argparse.ArgumentParser(description='Structured XLSR-based MDD training pipeline')
    parser.add_argument('--mode', choices=['wl', 'mfa'], default='wl', help='Training mode')
    parser.add_argument('--train_csv', type=str, default='train_time.csv', help='Path to train csv file')
    parser.add_argument('--dev_csv', type=str, default='dev_time.csv', help='Path to dev csv file')
    parser.add_argument('--train_wav_dir', type=str, default='/home/user14/trungnt/mdd/EN_MDD/WAV', help='Directory containing wav files')
    parser.add_argument('--dev_wav_dir', type=str, default='/home/user14/trungnt/mdd/EN_MDD/WAV', help='Directory containing wav files')
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to vocab json')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/l2_arctic', help='Output checkpoint directory')
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-base-100h')
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--eval_start_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    return parser.parse_args()

def infer():
    args = build_args()
    trainer = MDDTrainer(args)
    INFER_CSV = "test_time.csv"
    INFER_WAV_DIR = "/home/user14/trungnt/mdd/EN_MDD/WAV"
    trainer.inference(csv_path=INFER_CSV, wav_dir=INFER_WAV_DIR, batch_size=4, checkpoint="checkpoint/checkpoint_wl.pth")


def main():
    args = build_args()
    trainer = MDDTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()


