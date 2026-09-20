## Purpose

Given a test photo, decide whether the face in it matches the registered person and report a verdict with a similarity score.

## ADDED Requirements

### Requirement: Verify a test photo against the registration

The system SHALL compare a test photo's face against the registered reference and SHALL report a match/no-match verdict together with a similarity score.

#### Scenario: Test photo matches

- **WHEN** the test photo is the registered person and the similarity is at or above the threshold
- **THEN** the system reports a match together with the similarity score

#### Scenario: Test photo does not match

- **WHEN** the test photo is a different person and the similarity is below the threshold
- **THEN** the system reports no match together with the similarity score

### Requirement: Report a test photo with no detectable face

The system SHALL report an error when the test photo contains no detectable face.

#### Scenario: No face in test photo

- **WHEN** the user tests a photo that contains no detectable face
- **THEN** the system reports that no face was detected and produces no verdict

### Requirement: Handle multiple faces in a test photo

The system SHALL handle a test photo containing multiple faces by selecting the largest face and continuing verification against it.

#### Scenario: Multiple faces in test photo

- **WHEN** the test photo contains more than one detectable face
- **THEN** the system selects the largest face and verifies against it

### Requirement: Reject verification without a registration

The system SHALL report an error when verification is attempted while no person is registered.

#### Scenario: Test attempted before registration

- **WHEN** the user attempts to test a photo while no person is registered
- **THEN** the system reports that a person must be registered first and produces no verdict

### Requirement: Adjust the similarity threshold

The system SHALL allow the user to adjust the similarity threshold used to decide match/no-match.

#### Scenario: Change the threshold

- **WHEN** the user changes the threshold
- **THEN** subsequent verifications use the new threshold to produce their verdict
