"""
Class to step through a grid of weight points

History of changes:
Version 1.0
- Created 25th January 2011

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

import numpy as np

class GSSplineWeights:
    def __init__(self):
	self.nKnots		= 0
	self.nWeights		= 0
	self.maxStepDist	= 0
	self.CurrentShape	= 0
	self.nTotalShapes	= 1

	self.fKnots		= []
	self.fMaxKnots		= []
	self.fMinKnots		= []

	self.fMinEBL		= []
	self.fMaxEBL		= []
	self.fLogEBLSteps	= []
	return
#-------------------------------------------------------------------------#
    def ResetKnots(self):
	"""
	Set Knots to Minimum Values
	"""
	self.fKnots = [self.fMinKnots[i] for i in range(self.nKnots)]
	return
#-INITIAL VALUES----------------------------------------------------------#
#-------------------------------------------------------------------------#
    def FillInitWeights(self,MinEBL,MaxEBL,StiffGrid=False,LogSteps=[],GridMin=0., GridMax=0., \
    nWeights=12, nKnots = 16):
	"""
	Set initial weight and knot values 
	"""
	if not len(MinEBL) == len(MaxEBL):
	    raise TypeError("Lists containing min EBL and max EBL must have same length")
	if not len(MinEBL) == nKnots:
	    raise TypeError("Min and Max EBL Lists must have same length as nKnots")

# Hier Knoten statt MINEBL und MAXEBL!!!
	if not int(GridMin) and not int(GridMax):
	    self.fMinEBL	= np.array(MinEBL)
	    self.fMaxEBL	= np.array(MaxEBL)
	    self.fMinKnots	= np.zeros( (self.nKnots,), dtype=np.int )
	    self.fMaxKnots	= np.array( [ self.nWeights - 1 for i in range(self.nKnots) ] )
	else:
	    self.fMinEBL	= np.array( [ GridMin for i in range(nKnots) ] )
	    self.fMaxEBL	= np.array( [ GridMax for i in range(nKnots) ] )
	    self.fMinKnots	= np.array(MinEBL)
	    self.fMaxKnots	= np.array(MaxEBL)

	for i,k in enumerate(self.fMinKnots):
	    self.nTotalShapes *= (self.fMaxKnots[i] - k + 1)

	self.fCurrentWeights = self.fMinEBL[:]

	self.nKnots	= len(MaxEBL)

	self.nWeights	= nWeights
	self.nKnots	= nKnots

	if len(LogSteps) and len(LogSteps) == len(self.fMaxEBL):
	    self.fLogEBLSteps = np.array(LogSteps)
	else:
	    self.fLogEBLSteps = np.array( (np.log10(self.fMaxEBL) - np.log10(self.fMinEBL) ) \
		/ float(self.nWeights - 1) )

	# STIFF GRID between GridMin  and Grid Max :
	if StiffGrid:
	    self.fLogEBLSteps = np.array( [ ( np.log10(GridMax) - np.log10(GridMin) ) \
	    / (float(self.nWeights) - 1.) for i in range(self.nKnots) ] )


# set min knots
#	    for i in range(self.nKnots):
#		for j in range(self.nWeights):
#		    val = np.power(10.,np.log10(GridMin) + self.fLogEBLSteps[i]*float(j)) 
#		    if  val < self.fMinEBL[i]:
#			self.fMinKnots[i] = j+1
#		    else:
#			self.fMinKnots[i] = j
#			break
#		self.fCurrentWeights[i] = np.power(10.,np.log10(GridMin) + self.fLogEBLSteps[i]* \
#		float(self.fMinKnots[i]))
#
# set max knots
#	    for i in range(self.nKnots):
#		for j in range(self.nWeights-1,-1,-1):
#		    val = np.power(10.,np.log10(GridMin) + self.fLogEBLSteps[i]*float(j)) 
#		    if val > self.fMaxEBL[i]:
#			self.fMaxKnots[i] = j-1
#		    else:
#			self.fMaxKnots[i] = j
#			break

#	    self.fMinEBL = np.array ( [ GridMin for i in range(self.nKnots) ] )

	#self.fMaxKnots[0] += 1
	self.ResetKnots()
	self.maxStepDist = max(self.fMaxKnots) - min(self.fMinKnots)


	return
#-------------------------------------------------------------------------#
    def SetWeights(self):
	"""
	Set new weights with current set of knots
	"""
	self.fCurrentWeights = [np.power(10.,np.log10(self.fMinEBL[i]) \
	    + float(self.fKnots[i])*self.fLogEBLSteps[i]) \
	    for i in range(self.nKnots) ]
	return
#-------------------------------------------------------------------------#
    def SetKnots(self,knot):
	"""
	Increase current knots
	"""

	self.fKnots[knot] += 1
	if knot == 0:
	    if self.fKnots[0] > self.fMaxKnots[knot] - 1:
		return False
	    else:
		return True

	stepDist = self.fKnots[knot] - self.fKnots[knot - 1]
	stepDist = np.abs(stepDist)

	if self.fKnots[knot] > self.fMaxKnots[knot] or stepDist > self.maxStepDist:
	    returnVal = self.SetKnots(knot - 1)
	    newKnotVal = self.fKnots[knot - 1] - self.maxStepDist
	    if newKnotVal < self.fMinKnots[knot]:
		newKnotVal = self.fMinKnots[knot]
	    self.fKnots[knot] = newKnotVal
	    return returnVal

	return True
#-------------------------------------------------------------------------#
    def NextShape(self):
	"""
	Set next shape of Grid
	"""

	returnVal = self.SetKnots(self.nKnots - 1)
	if returnVal:
	    self.SetWeights()
	    self.CurrentShape += 1
	else:
	    self.CurrentShape = 0
	return returnVal
#-------------------------------------------------------------------------#
    def SetMaxKnots(self,MaxKnots=[]):
	"""
	Set Knots to maximum value allwoed by fMaxKnots or MaxKnots
	"""
	if not len(MaxKnots) == self.nKnots:
	    for i,k in enumerate(self.fMaxKnots):
		self.fKnots[i] = k
	else:
	    for i,k in enumerate(MaxKnots):
		self.fKnots[i] = k
	self.SetWeights()
	return True
#-------------------------------------------------------------------------#
    def SetMinKnots(self,MinKnots=[]):
	"""
	Set Knots to minimum value allwoed by fMinKnots or MinKnots
	"""
	if not len(MinKnots) == self.nKnots:
	    for i,k in enumerate(self.fMinKnots):
		self.fKnots[i] = k
	else:
	    for i,k in enumerate(MinKnots):
		self.fKnots[i] = k
	self.SetWeights()
	return True
#-------------------------------------------------------------------------#
    def SetGrid2CurrentShape(self,CurrentShape):
	""" 
	Set Grid to the Knots and weights that correspond to CurrentShape
	"""
	if CurrentShape >= self.nTotalShapes or CurrentShape < 0:
	    print "Requested Shape does not exist"
	    return False

	Currentknots = np.zeros(self.nKnots,np.int)

	def product(j,IncVec):
	    result = 1
	    if j + 1 > self.nKnots - 1:
		return 1
	    for i in range(j+1,self.nKnots):
		result *= (IncVec[i] + 1)
	    return result

	for j in range(0,self.nKnots-1):
	    prod = product(j,self.fMaxKnots-self.fMinKnots)
	    if CurrentShape + 1 >= prod:
		Currentknots[j]= self.fMinKnots[j] + (CurrentShape) / prod
		CurrentShape %= prod
	    else:
		Currentknots[j] = self.fMinKnots[j]
	Currentknots[self.nKnots - 1] = CurrentShape + self.fMinKnots[self.nKnots - 1]

	self.SetMaxKnots(Currentknots)
    
	return True
