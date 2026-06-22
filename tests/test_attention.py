import pytest

from backend.attention import AttentionTracker, Calibration, Observation


def obs(x=100, yaw=0, pitch=0, gaze_x=0, gaze_y=0):
    return Observation((x, 100, 100, 120), yaw, pitch, gaze_x, gaze_y)


def test_three_second_calibration_averages_samples():
    calibration = Calibration()
    calibration.start(now=10.0)
    calibration.add((2.0, 4.0, 0.1, -0.1), now=11.0)
    calibration.add((4.0, 6.0, 0.2, 0.1), now=13.0)
    assert calibration.calibrated
    assert calibration.center == pytest.approx((3.0, 5.0, 0.15, 0.0))


def test_calibration_times_out_cleanly_without_a_face():
    calibration = Calibration()
    calibration.start(now=2.0)
    calibration.tick(now=5.1)
    assert not calibration.active
    assert not calibration.calibrated


def test_attention_hysteresis_and_loss_cleanup():
    calibration = Calibration()
    tracker = AttentionTracker()
    tracker.update([obs()], calibration, now=0.0)
    assert not tracker.tracks[1].attentive
    tracker.update([obs()], calibration, now=0.6)
    tracker.update([obs()], calibration, now=1.2)
    assert tracker.tracks[1].attentive
    tracker.update([obs(yaw=40)], calibration, now=1.3)
    tracker.update([obs(yaw=40)], calibration, now=2.2)
    assert not tracker.tracks[1].attentive
    tracker.update([], calibration, now=3.3)
    assert not tracker.tracks


def test_two_people_receive_ephemeral_ids():
    tracker = AttentionTracker()
    calibration = Calibration()
    tracks = tracker.update([obs(50), obs(400)], calibration, now=1.0)
    assert [track.id for track in tracks] == [1, 2]
