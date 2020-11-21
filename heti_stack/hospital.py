import random

# Base hospital class

class Hospital(object):
	"""Hospitals are assumed to have one property that counts in the algorithm:
	- Capacity
	Further pathways like DRA can also be implemented"""
	def __init__(self, name: str, abbreviation: str, capacity: int, firsts: int, dra: int, remove_dra=False) -> None:
		self.name = name
		self.abbreviation = abbreviation
		self.is_dra = dra > 0
		self.dra = dra
		self.capacity = capacity if not remove_dra else capacity - dra
		self.firsts = firsts
		self.spots_remaining = capacity
		self.filled_spots = []
	def __repr__(self) -> str:
		return self.__str__()
	def __str__(self) -> str:
		return "'{name}' ({abbr}): {filled}/{capacity}".format(name=self.name, abbr=self.abbreviation,
	   		filled=self.capacity - self.spots_remaining, capacity=self.capacity)
	def as_dict(self) -> dict:
		return {
			"name": self.name,
			"abbreviation": self.abbreviation,
			"is_dra": self.is_dra,
			"dra": self.dra,
			"capacity": self.capacity,
			"firsts": self.firsts,
			"spots_remaining": self.spots_remaining,
			"filled_spots": self.filled_spots
		}
	def fill(self, applicants, dra_prefill=False) -> None:
		if len(applicants) > self.spots_remaining:
			selected_applicants = random.sample(applicants, self.spots_remaining)
		else:
			selected_applicants = applicants
		self.filled_spots += selected_applicants
		self.spots_remaining -= len(selected_applicants)
		for a in self.filled_spots:
			a.allocate(self, dra_prefill)
	def empty(self) -> None:
		while self.filled_spots:
			applicant = self.filled_spots.pop()
			applicant.free()
			self.spots_remaining += 1

# Class-dependent constants

hospitals = []
with open("hospital-networks.txt", "r") as hospital_infile:
	for line in hospital_infile:
		hname, hshort, hcap, hfirsts, hdra = line.split('\t')
		hospitals.append(Hospital(hname, hshort, int(hcap), int(hfirsts), int(hdra)))
hospital_weights = [h.firsts/366 for h in hospitals]
hospitals_with_weights = list(zip(hospitals, hospital_weights))

stack = []
with open("stack.txt", "r") as stack_infile:
	for line in stack_infile:
		stack.append(next((i for i, h in enumerate(hospitals) if h.abbreviation == line.strip()), None))
tier_one_hospitals = stack[:4]
top_six_hospitals = stack[:6]

altstack = []
with open("altstack.txt", "r") as altstack_infile:
	for line in altstack_infile:
		altstack.append(next((h for h in hospitals if h.abbreviation == line.strip()), None))
