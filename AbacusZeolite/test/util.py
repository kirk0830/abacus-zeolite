import sys

def init():
    '''initialize the workflow, return True if only test mode is on
        
    Unluckily, the argparse will be overwritten by the unittest, so we use
    sys.argv to pass the arguments to the unittest.
    '''
    if '--only-test-mode=true' in sys.argv:
        sys.argv.remove('--only-test-mode=true') # because we don't want to pass this to unittest
        return True
    return False

