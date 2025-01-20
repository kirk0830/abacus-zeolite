'''structure manipulation functions'''

# built-in modules
import os
import unittest
import logging
import itertools as it
import time
from multiprocessing import Process, Queue

# third-party modules
import numpy as np
from scipy.optimize import minimize

# home-made modules
from AbacusZeolite.test.util import init as test_init

class OctreeBox:
    '''Octree box for the zeolite structure
    
    Documentation
    -------------
    https://ucoyxk075n.feishu.cn/docx/WP8sdHJARoYrRExMbvMcvmFLnaf
    '''
    def __init__(self, root, code):
        '''
        build a Octree box
        
        Parameters
        ----------
        root : np.ndarray
            the root coordinate, always to be (0,0,0), then the octree
            will divide the space into 8 parts.
            
        a : float
            the length of the box, always to be 1
        
        code : str
            the code of the box, for example if the space goes in x/y/z
            direction in range [-0.5, 0.5), then the codes of boxes 
            divied at the first time are `---`, `++-`, `+-+`, `+--`, 
            `--+`, `+-+`, `++-`, `---`. Once the division goes on, there
            will be 
            `---,---`, 
            `---,--+`, `---,-+-`, `---,+--`,
            `---,++-`, `---,+-+`, `---,-++`,
            `---,+++`
            for `---` box. 
        '''
        self.root = np.array(root).reshape(3,)
        self.code = code
    
    def code2num(code: str):
        '''convert the code to the number (the displacement) by the rule:
        "-" -> -1, "+" -> 1
        . Comma sperated codes will be converted to differet lists
        '''
        # corner case: root box, code == ''
        if code == '':
            return np.array([0, 0, 0])
        
        code = code.split(',')
        return [np.array([1 if c == '+' else -1 for c in codei]) for codei in code]
    
    def num2code(num: np.ndarray):
        '''convert the number to the code by the rule:
        -1 -> -, 1 -> +
        . Different lists will be converted to comma separated codes
        '''
        # corner case: root box, code == ''
        if np.all(num == 0):
            return ''
        
        return ','.join([''.join(['+' if n == 1 else '-' for n in numi]) for numi in num])

    def center(self):
        '''get the center of the box'''
        # corner case: root box, code == ''
        if self.code == '':
            return self.root
        
        num = OctreeBox.code2num(self.code)
        disp = np.array([0., 0., 0.]).reshape(3,)
        for i, n in enumerate(num):
            disp += n*0.5**(i+2)
        return self.root + disp 

    def length(self):
        '''get the length of the box'''
        # corner case: root box, code == ''
        return 1 if self.code == '' else 0.5**(len(self.code.split(',')))

    def boundaries(self):
        '''get the boundaries of the box in format of xlo, xhi, ylo, yhi, zlo, zhi'''
        # corner case: root box, code == ''
        if self.code == '':
            return -0.5, 0.5, -0.5, 0.5, -0.5, 0.5
        
        center = self.center()
        length = self.length()
        return center[0] - length/2, center[0] + length/2, \
               center[1] - length/2, center[1] + length/2, \
               center[2] - length/2, center[2] + length/2

    def divide(self):
        '''divide the box into 8 parts'''
        # corner case: root box, code == ''
        if self.code == '':
            return [OctreeBox(self.root, ''.join(c)) 
                    for c in it.product(['-', '+'], repeat=3)]
        else:
            return [OctreeBox(self.root, ','.join([self.code, ''.join(c)]))
                    for c in it.product(['-', '+'], repeat=3)]

    def volume(self):
        '''get the volume of the box'''
        return self.length()**3

    @staticmethod
    def count_nat_in_box(atoms: np.ndarray,
                         xlo, xhi, ylo, yhi, zlo, zhi) -> int:
        '''calculate the number of atoms in the box'''
        return np.sum((atoms[:, 0] >= xlo) & (atoms[:, 0] < xhi) &
                      (atoms[:, 1] >= ylo) & (atoms[:, 1] < yhi) &
                      (atoms[:, 2] >= zlo) & (atoms[:, 2] < zhi))

    @staticmethod
    def is_adjacent(box1: 'OctreeBox', box2: 'OctreeBox'):
        '''check whether the other box is adjacent to the current box by
        calculating centers of them two and the length of them. If there
        is one component of the distance of the centers is equal to the
        sum of the length of them, the other two components should be 
        smaller than the sum of the length of them.
        '''
        center1, center2 = box1.center(), box2.center()
        length1, length2 = box1.length(), box2.length()
        
        dist = np.abs((center1 - center2 + 0.5) % 1 - 0.5) # we consider the PBC

        delta = np.array([((length1 + length2)/2 - d) for d in dist])
        # there should be one zero and two positive numbers
        return np.sum(delta == 0) == 1 and np.sum(delta > 0) == 2

    def adjacent_with(self, other: 'OctreeBox'):
        '''check whether the other box is adjacent to the current box'''
        return OctreeBox.is_adjacent(self, other)

    @staticmethod
    def boxgen(atoms: np.ndarray, size_thr = 1/2**4):
        '''generate the boxes in which there are no atoms in them. This
        function is useful when finding the cavities of the zeolite.
        
        Parameters
        ----------
        atoms : np.ndarray
            the atomic coordinates in shape (n, 3), should be given in
            direct coordinates, all components should be within the range
            [-0.5, 0.5)
        cell : np.ndarray
            the cell matrix in shape (3, 3), a flattened explanation would
            be [a11, a12, a13, a21, a22, a23, a31, a32, a33]
        size_thr : float
            the threshold of the box size. When the size of the box is
            smaller than this threshold, we stop the division.
        
        Returns
        -------
        list
            the list of the OctreeBox instances in which there are no atoms in them
        '''
        taud = atoms
        
        # initialize the iteration
        temp, boxes = [OctreeBox(np.array([0, 0, 0]), '')], []
        size_min, iter = 1, 1
        while size_min >= size_thr:
            # when the size of the box is smaller than the threshold, we stop
            logging.info(f'''iteration-{iter}
    size_min: {size_min}, 
    size_thr: {size_thr},
    number of boxes: {len(boxes)}
    ''')
            # if there is no atoms in the box, we add it to the list
            boxes.extend([box for box in temp 
                            if OctreeBox.count_nat_in_box(taud, *box.boundaries()) == 0])
            # for the rest, we divide them
            temp = [box for box in temp 
                    if OctreeBox.count_nat_in_box(taud, *box.boundaries()) > 0]
            if len(temp) == 0:
                break
            temp = [box for b in temp for box in b.divide()]
            size_min = min([box.length() for box in temp])

            iter += 1
            
        # summary
        logging.info(f'{"Box":<6}{"Center":<30}{"Length":<10}')
        logging.info('-'*(6+30+10))
        for i, box in enumerate(boxes):
            logging.info(f'{i:<6}{str(box.center()):<30}{box.length():<10}')
        
        return boxes

    @staticmethod
    def kmeans(boxes: list, ncluster: int, niter: int, tol: float):
        '''use the specialized version of K-means clustering algorithm
        to cluster the OctreeBoxes into n clusters, according to their
        PBC distances'''
        logging.info('Performing K-means clustering on OctreeBoxes >>')
        
        #centers = np.array([box.center() for box in np.random.choice(boxes, ncluster, replace=False)])
        # we use PCA to initialize the centers
        centers = np.array([box.center() for box in np.random.choice(boxes, ncluster, replace=False)])
        
        clusters = [[] for _ in range(ncluster)]
        
        logging.info(f'{"Iter":<6}{"Norm":<10}')
        logging.info('-'*(6+10))
                    
        for i in range(niter):
            clusters = [[] for _ in range(ncluster)] # reset
            for box in boxes:
                dist = np.linalg.norm((centers - box.center() + 0.5)%1 - 0.5, axis=1)
                clusters[np.argmin(dist)].append(box) # box assigned to the nearest center

            # update the centers: average coordinate of boxes' centers of the cluster
            # BUG: the average is not directly the center of the cluster, instead, the
            #      average between a and b should be:
            #      ((a%1 + b%1)/2 - 0.5)%1 + 0.5
            #      for example the -0.4 and +0.4, the correct answer should be +/-0.5
            centers_new = np.array([(np.average(np.array([box.center() for box in cluster])%1, axis=0) - 0.5)%1 + 0.5
                                    for cluster in clusters])
            
            eps = np.linalg.norm(centers_new%1) - np.linalg.norm(centers%1)
            logging.info(f'{i:<6}{eps:<10}')
            
            if np.abs(eps) <= tol:
                break
            
            centers = centers_new
            
        logging.info('<< K-means clustering finished')
        return centers, clusters

    @staticmethod
    def sklearn_cluster_method(boxes: list):
        '''use the support vector machine to classify the boxes with the distance
        defined by PBC adpated distance'''
        from sklearn.cluster import AffinityPropagation
        from sklearn.metrics import pairwise_distances
        
        data = np.array([box.center() for box in boxes] )
        # customized definition of the distance
        def eval_dist(x, y):
            return np.linalg.norm((x - y + 0.5)%1 - 0.5)
        
        clustering = AffinityPropagation(affinity='precomputed',
                                         verbose=True,
                                         preference=-1)
        mat = pairwise_distances(data, metric=eval_dist)
        
        labels = clustering.fit_predict(mat)
        
        # returns the center of each cluster
        centers = []
        for i in np.unique(labels):
            centers.append(np.average(data[labels == i], axis=0))
                    
        return centers            

def spring_relax(cell: np.ndarray, 
                         tau: np.ndarray, 
                         spacing=0.25, 
                         direct=True,
                         optimizer='Powell') -> np.ndarray:
    '''
    find the cavities of the zeolite, return the coordinates of the
    center.
    
    Algorithm
    ---------
    For zeolite, the center of the cavity is always the high-symmetry
    points of the zeolite. For any perturbation which drives the point
    deviates from the high-symmetry point, will bring the raise of the
    sum of squared distance to all atoms. Therefore, we can use 
    optimization algorithm to find the center of the cavity.
    
    Parameters
    ----------
    cell : np.ndarray
        the cell matrix in shape (3, 3), a flattened explanation would
        be [a11, a12, a13, a21, a22, a23, a31, a32, a33]
    tau : np.ndarray
        the atomic coordinates in shape (n, 3). If the `direct` is True,
        it should be the direct coordinates, otherwise, it should be the
        Cartesian coordinates.
    spacing : float
        the spacing between the grid points. The smaller the spacing, the
        more accurate the result will be, but the more time it will cost.
        The unit of it will be assumed to be the same as the unit of the
        cell matrix.
    direct : bool
        whether the coordinates are in direct coordinates or not (default
        is True). The return will also depend on this parameter.
        
    Returns
    -------
    np.ndarray
        the coordinates of the cavity center in shape (m, 3). If the
        `direct` is True, the coordinates will be in direct coordinates,
        otherwise, the coordinates will be in Cartesian coordinates.
    '''
    # we will use direct coordinates in the following
    taud = tau if direct else np.linalg.solve(cell.T, tau.T).T
    nx, ny, nz = [int(np.ceil(np.linalg.norm(cell[i]) / spacing)) for i in range(3)]
    
    # ranging from 0 to 1 in each direction
    seed = np.array(list(it.product(np.arange(nx), np.arange(ny), np.arange(nz)))) / np.array([nx, ny, nz])
    logging.info(f'number of seeds: {len(seed)}')
    
    def _obj(x): # remember to consider the PBC when calculating the distance...
        return np.sum([np.linalg.norm(((x - t + 0.5)%1 - 0.5)@cell.T)**2 for t in taud])
    def _opt(s: np.ndarray, q: Queue):
        res = minimize(_obj, s, method=optimizer, tol=1e-5)
        q.put(res.x)
    
    center = []
    nproc = os.cpu_count() # get the number of processors
    logging.info(f'number of processors: {nproc}')
    ntask = len(seed)
    while ntask > 0:
        # each time we parallelize `nproc` optimization tasks
        i = len(seed) - ntask # the starting index
        
        queue = Queue()
        tasks = [Process(target=_opt, args=(seed[i+j], queue)) 
                 for j in range(min(nproc, ntask))]
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()
            center.append(queue.get())
            ntask -= 1

        logging.info(f'{ntask} tasks left')
    
    # delete the center with duplicated distance
    norm = [_obj(c) for c in center]
    idx = np.unique(norm, return_index=True)[1]
    center = np.array(center)[idx]

    return center if direct else center @ cell

class TestOctreeBox(unittest.TestCase):
    def test_code_and_num(self):
        code = '---,---'
        self.assertEqual(OctreeBox.num2code(OctreeBox.code2num(code)), code)
        code = '---,--+'
        self.assertEqual(OctreeBox.num2code(OctreeBox.code2num(code)), code)
        code = '---,-+-'
        self.assertEqual(OctreeBox.num2code(OctreeBox.code2num(code)), code)
        code = '+-+,++-,---'
        self.assertEqual(OctreeBox.num2code(OctreeBox.code2num(code)), code)

    def test_center(self):
        # order 1
        box = OctreeBox(root=[0, 0, 0], code='---')
        self.assertTrue(np.allclose(box.center(), [-1/4, -1/4, -1/4]))
        box = OctreeBox(root=[0, 0, 0], code='--+')
        self.assertTrue(np.allclose(box.center(), [-1/4, -1/4, 1/4]))
        box = OctreeBox(root=[0, 0, 0], code='+-+')
        self.assertTrue(np.allclose(box.center(), [1/4, -1/4, 1/4]))
        # order 2
        box = OctreeBox(root=[0, 0, 0], code='---,---')
        self.assertTrue(np.allclose(box.center(), [-3/8, -3/8, -3/8]))
        box = OctreeBox(root=[0, 0, 0], code='---,--+')
        self.assertTrue(np.allclose(box.center(), [-3/8, -3/8, -1/8]))
        box = OctreeBox(root=[0, 0, 0], code='-+-,-++')
        self.assertTrue(np.allclose(box.center(), [-3/8, 3/8, -1/8]))
    
    def test_length(self):
        # order 1
        box = OctreeBox(root=[0, 0, 0], code='---')
        self.assertEqual(box.length(), 1/2)
        box = OctreeBox(root=[0, 0, 0], code='--+')
        self.assertEqual(box.length(), 1/2)
        box = OctreeBox(root=[0, 0, 0], code='+-+')
        self.assertEqual(box.length(), 1/2)
        # order 2
        box = OctreeBox(root=[0, 0, 0], code='---,---')
        self.assertEqual(box.length(), 1/4)
        box = OctreeBox(root=[0, 0, 0], code='---,--+')
        self.assertEqual(box.length(), 1/4)
        box = OctreeBox(root=[0, 0, 0], code='-+-,-++')
        self.assertEqual(box.length(), 1/4)
        
    def test_boundaries(self):
        # order 1
        box = OctreeBox(root=[0, 0, 0], code='---')
        self.assertTrue(np.allclose(box.boundaries(), [-1/2, 0, -1/2, 0, -1/2, 0]))
        box = OctreeBox(root=[0, 0, 0], code='--+')
        self.assertTrue(np.allclose(box.boundaries(), [-1/2, 0, -1/2, 0, 0, 1/2]))
        box = OctreeBox(root=[0, 0, 0], code='+-+')
        self.assertTrue(np.allclose(box.boundaries(), [0, 1/2, -1/2, 0, 0, 1/2]))
        # order 2
        box = OctreeBox(root=[0, 0, 0], code='---,---')
        self.assertTrue(np.allclose(box.boundaries(), [-1/2, -1/4, -1/2, -1/4, -1/2, -1/4]))
        box = OctreeBox(root=[0, 0, 0], code='---,--+')
        self.assertTrue(np.allclose(box.boundaries(), [-1/2, -1/4, -1/2, -1/4, -1/4, 0]))
        box = OctreeBox(root=[0, 0, 0], code='-+-,-++')
        self.assertTrue(np.allclose(box.boundaries(), [-1/2, -1/4, 1/4, 1/2, -1/4, 0]))

    def test_divide(self):
        # order 1
        box = OctreeBox(root=[0, 0, 0], code='---')
        boxes = box.divide()
        self.assertEqual(len(boxes), 8)
        codes = [b.code for b in boxes]
        for c in it.product(['-', '+'], repeat=3):
            self.assertIn(','.join([box.code, ''.join(c)]), codes)
        # order 2
        box = OctreeBox(root=[0, 0, 0], code='---,---')
        boxes = box.divide()
        self.assertEqual(len(boxes), 8)
        codes = [b.code for b in boxes]
        for c in it.product(['-', '+'], repeat=3):
            self.assertIn(','.join([box.code, ''.join(c)]), codes)

    def test_volume(self):
        # order 1
        box = OctreeBox(root=[0, 0, 0], code='---')
        self.assertEqual(box.volume(), 1/8)
        box = OctreeBox(root=[0, 0, 0], code='--+')
        self.assertEqual(box.volume(), 1/8)
        box = OctreeBox(root=[0, 0, 0], code='+-+')
        self.assertEqual(box.volume(), 1/8)
        # order 2
        box = OctreeBox(root=[0, 0, 0], code='---,---')
        self.assertEqual(box.volume(), 1/64)
        box = OctreeBox(root=[0, 0, 0], code='---,--+')
        self.assertEqual(box.volume(), 1/64)
        box = OctreeBox(root=[0, 0, 0], code='-+-,-++')
        self.assertEqual(box.volume(), 1/64)

    def test_is_adjacent(self):
        # order 1
        box1 = OctreeBox(root=[0, 0, 0], code='---')
        box2 = OctreeBox(root=[0, 0, 0], code='--+')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box2 = OctreeBox(root=[0, 0, 0], code='+-+')
        self.assertFalse(OctreeBox.is_adjacent(box1, box2))
        # order 2
        box1 = OctreeBox(root=[0, 0, 0], code='---,---')
        box2 = OctreeBox(root=[0, 0, 0], code='---,--+')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box2 = OctreeBox(root=[0, 0, 0], code='---,-+-')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box2 = OctreeBox(root=[0, 0, 0], code='---,+-+')
        self.assertFalse(OctreeBox.is_adjacent(box1, box2))
        box2 = OctreeBox(root=[0, 0, 0], code='---,-++')
        self.assertFalse(OctreeBox.is_adjacent(box1, box2))
        # order 2
        box1 = OctreeBox(root=[0, 0, 0], code='---,-++')
        box2 = OctreeBox(root=[0, 0, 0], code='-+-,--+')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box1 = OctreeBox(root=[0, 0, 0], code='---,+++')
        box2 = OctreeBox(root=[0, 0, 0], code='--+,++-')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box1 = OctreeBox(root=[0, 0, 0], code='---,+++')
        box2 = OctreeBox(root=[0, 0, 0], code='---,+++')
        self.assertFalse(OctreeBox.is_adjacent(box1, box2)) # overlapped!
        # different order
        box1 = OctreeBox(root=[0, 0, 0], code='---')
        box2 = OctreeBox(root=[0, 0, 0], code='--+,++-')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box2 = OctreeBox(root=[0, 0, 0], code='--+,+++')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2)) # because PBC!
        box2 = OctreeBox(root=[0, 0, 0], code='-+-,+-+')
        self.assertTrue(OctreeBox.is_adjacent(box1, box2))
        box2 = OctreeBox(root=[0, 0, 0], code='+++,+++')
        self.assertFalse(OctreeBox.is_adjacent(box1, box2))

    def test_count_nat_in_box(self):
        atoms = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        xlo, xhi, ylo, yhi, zlo, zhi = 1, 3, 1, 3, 1, 3
        self.assertEqual(OctreeBox.count_nat_in_box(atoms, 
                        xlo, xhi, ylo, yhi, zlo, zhi), 2)
        xlo, xhi, ylo, yhi, zlo, zhi = 0, 3, 0, 3, 0, 3
        self.assertEqual(OctreeBox.count_nat_in_box(atoms, 
                        xlo, xhi, ylo, yhi, zlo, zhi), 3)
        xlo, xhi, ylo, yhi, zlo, zhi = 0, 1, 0, 1, 0, 1
        self.assertEqual(OctreeBox.count_nat_in_box(atoms, 
                        xlo, xhi, ylo, yhi, zlo, zhi), 1)
        xlo, xhi, ylo, yhi, zlo, zhi = 1, 2, 1, 2, 1, 2
        self.assertEqual(OctreeBox.count_nat_in_box(atoms, 
                        xlo, xhi, ylo, yhi, zlo, zhi), 1)

    @unittest.skip('now the boxes can be founded')
    def test_boxgen(self):
        '''this is a workflow, using the octree method to find all
        the cubes with no atoms in them, then we can find the cavity'''
        from AbacusZeolite.data.IZA import download
        from ase.io.cif import read_cif

        fn = download(name='MFI')
        parsed = read_cif(fn)
        taud = parsed.get_scaled_positions()

        # with in the range [-0.5, 0.5)
        taud = (taud + 0.5) % 1 - 0.5

        # get the boxes
        boxes = OctreeBox.boxgen(taud, 1/2**4)

        # boxes should only have the length of 1/2**n
        ln2 = np.log(2)
        for box in boxes:
            self.assertAlmostEqual(np.log(box.length())%ln2, 0, delta=1e-10)

    # @unittest.skip('Not yet implemented')
    def test_box_merge(self):
        from AbacusZeolite.data.IZA import download
        from ase.io.cif import read_cif
        from ase.io import write
        from ase import Atoms

        name = 'MFI'
        dr = 2.0 # in Angstrom
        
        fn = download(name=name)
        parsed = read_cif(fn)
        
        level = max([int(np.ceil(np.log(c/dr)/np.log(2))) for c in parsed.cell.cellpar()[:3]])
        
        tauc = parsed.get_positions()
        cell = parsed.get_cell() # in Angstrom
        elem = parsed.get_chemical_symbols()
        
        # with in the range [-0.5, 0.5)
        taud = (np.linalg.solve(cell.T, tauc.T).T + 0.5) % 1 - 0.5

        # there is a possible strategy for finding the centers of cavaties
        # 1. do the OctreeBox.boxgen to split the space into small boxes
        # 2. remove those boxes with the smallest size because those are
        #    the boxes with atoms in them
        # 3. unify the boxes with the same size by dividing them, this can
        #    unify the distance of the centers of the boxes
        # 4. do k-means clustering to find the centers of the cavaties
        
        # 1. do the OctreeBox.boxgen to split the space into small boxes
        boxes = OctreeBox.boxgen(taud, 1/2**level)
        
        # 2. remove those boxes with the smallest size because those are
        #    the boxes with atoms in them
        sizes = np.unique([box.length() for box in boxes])
        if len(sizes) > 1:
            min_size = min(sizes)
            boxes = [box for box in boxes if box.length() != min_size]
        # otherwise, all boxes with the same size, we do nothing
        
        # 3. unify the boxes with the same size by dividing them, this can
        #    unify the distance of the centers of the boxes
        boxes_unified = []
        for box in boxes:
            length = box.length()
            boxes_ = [box]
            while length > 1/2**level:
                boxes_ = [b for box_ in boxes_ for b in box_.divide()]
                length /= 2
            boxes_unified.extend(boxes_)
        
        # 4. do k-means clustering to find the centers of the cavaties
        centers, _ = OctreeBox.kmeans(boxes_unified, 
                                      20, 
                                      niter=1000, 
                                      tol=1e-10)
        # centers = OctreeBox.sklearn_cluster_method(boxes_unified)
        
        # write the centers to a new file
        boxes = [box.center() for box in boxes_unified]
        taud_new = np.concatenate([taud, centers, boxes])
        tauc_new = taud_new @ cell
        elem_new = elem + ['Rn'] * len(centers) + ['X'] * len(boxes)
        temp = Atoms(elem_new, positions=tauc_new, cell=cell)
        write(f'{name}_cavity.cif', temp)

class TestStructureZeoliteUtil(unittest.TestCase):
    
    @unittest.skip('too time-consuming, deprecated')
    def test_spring_relax(self):
        from AbacusZeolite.data.IZA import download
        from ase.io.cif import read_cif
        from ase.io import write
        from ase import Atoms
        
        fn = download(name='MFI')
        parsed = read_cif(fn)
        cell = parsed.get_cell()
        tauc = parsed.get_positions()
        elem = parsed.get_chemical_symbols()
        
        center = spring_relax(cell, tauc, spacing=2, direct=False,
                          optimizer='CG')
        # get the last 10 centers
        center = center[-10:]
        
        # write a new file with the center
        tauc_new = np.concatenate([tauc, center])
        elem_new = elem + ['X'] * len(center)
        temp = Atoms(elem_new, positions=tauc_new, cell=cell)
        write('test.cif', temp)


if __name__ == '__main__':
    flog = f'abacus-zeolite.structure.zeolite@{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=flog)
    exit_after_test = test_init()
    unittest.main(exit=exit_after_test)
    