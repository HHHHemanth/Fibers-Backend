MASKS = {}
DIST_MAPS = {}
FIBERS = {}
PARTICLE_MASKS = {}
PIXEL_SIZES = {}
def save_data(image_id, mask, dist_map, particle_mask):
    MASKS[image_id] = mask
    DIST_MAPS[image_id] = dist_map
    PARTICLE_MASKS[image_id] = particle_mask
    FIBERS[image_id] = []

def get_particle_mask(image_id):
    return PARTICLE_MASKS.get(image_id)

def get_mask(image_id):
    return MASKS.get(image_id)

def get_dist_map(image_id):
    return DIST_MAPS.get(image_id)

def save_fiber(image_id, fiber):
    FIBERS[image_id].append(fiber)

def get_fibers(image_id):
    return FIBERS.get(image_id, [])
def save_pixel_size(image_id, pixel_size):
    PIXEL_SIZES[image_id] = pixel_size

def get_pixel_size(image_id):
    return PIXEL_SIZES.get(image_id, 1.0)