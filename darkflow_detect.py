"""
Component that performs Darkflow object detection on images.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/image_processing.darkflow_detect/
"""
from datetime import timedelta
import logging

import requests
import voluptuous as vol

from homeassistant.components.image_processing import (
    CONF_ENTITY_ID, CONF_NAME, CONF_SOURCE, PLATFORM_SCHEMA,
    ImageProcessingEntity)
from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv

REQUIREMENTS = ['darkflow==1.0.0']

_LOGGER = logging.getLogger(__name__)

ATTR_MATCHES = 'matches'
ATTR_TOTAL_MATCHES = 'total_matches'

CONF_OPTIONS = 'options'
CONF_MODEL = 'detect_model'
CONF_WEIGHTS = 'weights'
CONF_LABELS = 'labels'
CONF_CONFIDENCE = 'confidence'
CONF_CROP = 'crop'
CONF_X1 = 'x1'
CONF_X2 = 'x2'
CONF_Y1 = 'y1'
CONF_Y2 = 'y2'

DEFAULT_TIMEOUT = 10
DEFAULT_CONFIDENCE = 0.55
DEFAULT_CROP = {'x1': None, 'x2': None, 'y1': None, 'y2': None}

SCAN_INTERVAL = timedelta(seconds=30)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_OPTIONS): {
        vol.Required(CONF_MODEL): cv.isfile,
        vol.Required(CONF_WEIGHTS): cv.isfile,
        vol.Required(CONF_LABELS): cv.isfile,
        vol.Optional(CONF_CROP, DEFAULT_CROP): {
            vol.Required(CONF_X1): vol.All(vol.Coerce(int), vol.Range(min=0)),
            vol.Required(CONF_Y1): vol.All(vol.Coerce(int), vol.Range(min=0)),
            vol.Required(CONF_X2): vol.All(vol.Coerce(int), vol.Range(min=0)),
            vol.Required(CONF_Y2): vol.All(vol.Coerce(int), vol.Range(min=0))
        },
        vol.Optional(CONF_CONFIDENCE, DEFAULT_CONFIDENCE):
            vol.All(vol.Coerce(float), vol.Range(
                min=0, max=1, max_included=True))
    }
})

def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up the Darkflow image processing platform."""
    try:
        # Verify that the Darkflow python package is pre-installed
        # pylint: disable=unused-import,unused-variable
        from darkflow.net.build import TFNet
    except ImportError:
        _LOGGER.error(
            "No Darkflow library found! Install or compile for your system "
            "following instructions here: https://github.com/thtrieu/darkflow")
        return
    options = {"model": config[CONF_OPTIONS][CONF_MODEL], "load": config[CONF_OPTIONS][CONF_WEIGHTS],
               "threshold": config[CONF_OPTIONS][CONF_CONFIDENCE], "labels": config[CONF_OPTIONS][CONF_LABELS]}
    entities = []
    if CONF_CROP not in config[CONF_OPTIONS].keys():
        config[CONF_OPTIONS][CONF_CROP] = False
    for camera in config[CONF_SOURCE]:
        entities.append(DarkflowImageProcessor(
            hass, camera[CONF_ENTITY_ID], camera.get(CONF_NAME), options, config[CONF_OPTIONS][CONF_CROP]))

    add_devices(entities)


class DarkflowImageProcessor(ImageProcessingEntity):
    """Representation of a Darkflow image processor."""

    def __init__(self, hass, camera_entity, name, options, crop):
        """Initialize the Darkflow entity."""
        from darkflow.net.build import TFNet
        self.hass = hass
        self._camera_entity = camera_entity
        if name:
            self._name = name
        else:
            self._name = "Darkflow {0}".format(
                split_entity_id(camera_entity)[1])
                
        self._matches = {}
        self._total_matches = 0
        self._last_image = None
        self._tfnet = TFNet(options)
        self._crop = crop

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def name(self):
        """Return the name of the image processor."""
        return self._name

    @property
    def state(self):
        """Return the state of the entity."""
        return self._total_matches

    @property
    def state_attributes(self):
        """Return device specific state attributes."""
        return {
            ATTR_MATCHES: self._matches,
            ATTR_TOTAL_MATCHES: self._total_matches
        }

    def process_image(self, image):
        """Process the image."""
        import numpy
        from PIL import Image
        import io

        fak_file = io.BytesIO(image)
        cv_image = numpy.array(Image.open(fak_file))
        
        # If there are parameters for cropping, split the image using numpy array splitting
        if self._crop:
            frame = cv_image[self._crop[CONF_Y1]:self._crop[CONF_Y2],
                             self._crop[CONF_X1]:self._crop[CONF_X2]]
        else:
            frame = cv_image
        object_locations = self._tfnet.return_predict(frame)

        found_object = 0
        matches = {}
        for det_object in object_locations:
            regions = []
            if det_object['label'] in matches.keys():
                regions = matches[det_object['label']]
                regions.append(((det_object['topleft']['x'], det_object['topleft']['y']), (
                    det_object['bottomright']['x'], det_object['bottomright']['y'])))
                found_object += 1
                matches[det_object['label']] = regions
            else:
                regions.append((int(det_object['topleft']['x']), int(det_object['topleft']['y']), int(
                    det_object['bottomright']['x']), int(det_object['bottomright']['y'])))
                found_object += 1
                matches[det_object['label']] = regions

        self._matches = matches
        self._total_matches = found_object
