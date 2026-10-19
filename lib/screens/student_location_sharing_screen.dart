import 'dart:async';

import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';

import '../api.dart';
import '../services/location_service.dart';
import '../theme.dart';

class StudentLocationSharingScreen extends StatefulWidget {
  const StudentLocationSharingScreen({super.key});

  @override
  State<StudentLocationSharingScreen> createState() =>
      _StudentLocationSharingScreenState();
}

class _StudentLocationSharingScreenState
    extends State<StudentLocationSharingScreen> {
  StreamSubscription<Position>? _positionSubscription;
  int? _sessionId;
  int _duration = 480;
  bool _historyEnabled = false;
  bool _geofencingEnabled = false;
  bool _loading = true;
  bool _saving = false;
  String? _message;

  @override
  void initState() {
    super.initState();
    _loadSession();
  }

  @override
  void dispose() {
    _positionSubscription?.cancel();
    super.dispose();
  }

  Future<void> _loadSession() async {
    try {
      final sessions = await Api().getStudentLocationSharingSessions();
      final now = DateTime.now();
      final active = sessions.whereType<Map>().map(Map<String, dynamic>.from).where((item) {
        final revoked = item['revoked_at'];
        final expires = DateTime.tryParse('${item['expires_at'] ?? ''}')?.toLocal();
        return revoked == null && expires != null && expires.isAfter(now);
      }).cast<Map<String, dynamic>>().firstOrNull;
      if (active != null) {
        _sessionId = int.tryParse('${active['id']}');
        _historyEnabled = active['history_enabled'] == true;
        _geofencingEnabled = active['geofencing_enabled'] == true;
        _startPositionUpdates();
      }
    } catch (_) {
      _message = 'Location sharing status could not be loaded.';
    }
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _startSharing() async {
    setState(() {
      _saving = true;
      _message = null;
    });
    try {
      final allowed = await LocationService.ensurePermission();
      if (!allowed) throw Exception('Allow location access to start sharing.');
      final response = await Api().startStudentLocationSharing(
        durationMinutes: _duration,
        historyEnabled: _historyEnabled,
        geofencingEnabled: _geofencingEnabled,
      );
      _sessionId = int.tryParse('${response['session']?['id']}');
      if (_sessionId == null) throw Exception('Location session could not start.');
      await _sendCurrentLocation();
      await _startPositionUpdates();
      if (mounted) setState(() => _message = 'Location sharing is active.');
    } catch (error) {
      if (mounted) {
        setState(() => _message = error.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _startPositionUpdates() async {
    await _positionSubscription?.cancel();
    if (_sessionId == null) return;
    _positionSubscription = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 25,
      ),
    ).listen((position) => _sendPosition(position));
  }

  Future<void> _sendCurrentLocation() async {
    final position = await LocationService.getCurrentPosition();
    if (position != null) await _sendPosition(position);
  }

  Future<void> _sendPosition(Position position) async {
    final sessionId = _sessionId;
    if (sessionId == null) return;
    try {
      await Api().sendStudentLocationUpdate(
        sessionId: sessionId,
        latitude: position.latitude,
        longitude: position.longitude,
        accuracyMeters: position.accuracy,
      );
      if (mounted) setState(() => _message = 'Last location sent just now.');
    } catch (_) {
      // A later GPS reading can recover from a temporary connectivity issue.
    }
  }

  Future<void> _stopSharing() async {
    final sessionId = _sessionId;
    if (sessionId == null) return;
    setState(() => _saving = true);
    try {
      await Api().revokeStudentLocationSharing(sessionId);
      await _positionSubscription?.cancel();
      _positionSubscription = null;
      if (mounted) {
        setState(() {
          _sessionId = null;
          _message = 'Location sharing stopped.';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _message = error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final active = _sessionId != null;
    return Scaffold(
      appBar: AppBar(title: const Text('Location sharing')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(20),
              children: [
                Icon(active ? Icons.location_on_rounded : Icons.location_off_rounded,
                    size: 64, color: active ? AppTheme.teal : Colors.grey),
                const SizedBox(height: 16),
                Text(
                  active ? 'You are sharing your location' : 'Share your location with your parent',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 10),
                const Text(
                  'Your linked parent can only see your location while this sharing session is active. You can stop it at any time.',
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 24),
                if (!active) ...[
                  DropdownButtonFormField<int>(
                    initialValue: _duration,
                    decoration: const InputDecoration(labelText: 'Share for'),
                    items: const [
                      DropdownMenuItem(value: 60, child: Text('1 hour')),
                      DropdownMenuItem(value: 480, child: Text('8 hours')),
                      DropdownMenuItem(value: 1440, child: Text('24 hours')),
                    ],
                    onChanged: _saving ? null : (value) => setState(() => _duration = value ?? 480),
                  ),
                  SwitchListTile(
                    value: _historyEnabled,
                    onChanged: _saving ? null : (value) => setState(() => _historyEnabled = value),
                    title: const Text('Share route history'),
                    subtitle: const Text('Allow the parent to view updates from this session.'),
                  ),
                  SwitchListTile(
                    value: _geofencingEnabled,
                    onChanged: _saving ? null : (value) => setState(() => _geofencingEnabled = value),
                    title: const Text('Allow geofence alerts'),
                    subtitle: const Text('Allow parent-created arrival and departure alerts.'),
                  ),
                ],
                if (_message != null) ...[
                  const SizedBox(height: 16),
                  Text(_message!, textAlign: TextAlign.center,
                      style: TextStyle(color: active ? AppTheme.teal : Colors.redAccent)),
                ],
                const SizedBox(height: 24),
                FilledButton.icon(
                  onPressed: _saving ? null : (active ? _stopSharing : _startSharing),
                  icon: Icon(active ? Icons.stop_circle_outlined : Icons.share_location_rounded),
                  label: Text(active ? 'Stop sharing' : 'Start sharing'),
                ),
              ],
            ),
    );
  }
}

extension _FirstOrNull<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
