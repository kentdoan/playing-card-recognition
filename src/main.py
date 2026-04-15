import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(PROJECT_ROOT, 'card_templates.npz')
TEST_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'test')
SUPPORTED_METHODS = ['template', 'sift', 'orb']
DEFAULT_METHOD = 'template'
DEFAULT_TEST_IMAGE = 'Cards_1.jpg'  #change if needed


def train_if_needed():
    if os.path.exists(TEMPLATE_FILE):
        return True

    try:
        train_script = os.path.join(PROJECT_ROOT, 'train.py')
        subprocess.run(['python', train_script], check=True, cwd=PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return False


def run_test(method, test_image):
    original_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        sys.path.insert(0, PROJECT_ROOT)

        import test
        test.test_image(test_image, method=method)
        return True
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(
        description='Playing card recognition from an input image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --method template
  python main.py --method sift --image 8cards_top.jpg
  python main.py --method orb
        """
    )
    
    parser.add_argument(
        '--method',
        choices=SUPPORTED_METHODS,
        default=DEFAULT_METHOD,
        help=f'Feature extraction method (default: {DEFAULT_METHOD})'
    )
    
    parser.add_argument(
        '--image',
        default=DEFAULT_TEST_IMAGE,
        help=f'Test image filename inside test folder (default: {DEFAULT_TEST_IMAGE})'
    )
    
    args = parser.parse_args()

    test_image_path = os.path.join(TEST_DIR, args.image)
    if not os.path.exists(test_image_path):
        print(f"Error: test image does not exist: {test_image_path}")
        sys.exit(1)

    if not train_if_needed():
        sys.exit(1)

    if not run_test(args.method, args.image):
        sys.exit(1)


if __name__ == '__main__':
    main()
