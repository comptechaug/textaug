#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

import pandas as pd
import re

with open("Sinonims.txt") as sFile:
	strWithAllFile = sFile.read()

print(strWithAllFile[:100])

