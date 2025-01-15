# This script is used to run unittests for the structure package
where_am_i=`dirname $0`
root=`cd $where_am_i/..; pwd`

#################
# run unittests #
#################

# structure module
python3 ${root}/structure/manip.py --only-test-mode=true
python3 ${root}/structure/util.py --only-test-mode=true
python3 ${root}/structure/surface.py --only-test-mode=true

# calculator module

# data module
