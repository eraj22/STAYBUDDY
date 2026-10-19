import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../../api.dart';
import '../../theme.dart';

class ParentLocationScreen extends StatefulWidget {
  const ParentLocationScreen({super.key});

  @override
  State<ParentLocationScreen> createState() => _ParentLocationScreenState();
}

class _ParentLocationScreenState extends State<ParentLocationScreen> {
  Timer? _refreshTimer;
  List<Map<String, dynamic>> _children = [];
  Map<String, dynamic>? _location;
  int? _studentId;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadChildren();
    _refreshTimer = Timer.periodic(const Duration(seconds: 30), (_) => _loadLocation());
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadChildren() async {
    try {
      final children = (await Api().getParentChildren())
          .whereType<Map>()
          .map((item) => Map<String, dynamic>.from(item))
          .toList();
      if (!mounted) return;
      setState(() {
        _children = children;
        _studentId ??= children.isEmpty ? null : int.tryParse('${children.first['id']}');
      });
      await _loadLocation();
    } catch (error) {
      if (mounted) setState(() { _error = error.toString().replaceFirst('Exception: ', ''); _loading = false; });
    }
  }

  Future<void> _loadLocation() async {
    final studentId = _studentId;
    if (studentId == null) {
      if (mounted) setState(() { _loading = false; _error = 'No linked student found.'; });
      return;
    }
    try {
      final data = await Api().getParentChildLocation(studentId);
      if (mounted) setState(() { _location = Map<String, dynamic>.from(data['location'] as Map); _error = null; _loading = false; });
    } catch (error) {
      if (mounted) setState(() { _location = null; _error = error.toString().replaceFirst('Exception: ', ''); _loading = false; });
    }
  }

  double _number(dynamic value) => double.tryParse('${value ?? 0}') ?? 0;

  @override
  Widget build(BuildContext context) {
    final latitude = _number(_location?['latitude']);
    final longitude = _number(_location?['longitude']);
    final hasLocation = _location != null;
    final selectedChild = _children.where((child) => '${child['id']}' == '$_studentId').firstOrNull;
    final childName = selectedChild?['name']?.toString() ?? 'Your child';
    return Scaffold(
      appBar: AppBar(title: const Text('Live location'), actions: [IconButton(onPressed: _loadLocation, icon: const Icon(Icons.refresh_rounded))]),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _loadLocation,
              child: ListView(padding: const EdgeInsets.all(16), children: [
                if (_children.length > 1) DropdownButtonFormField<int>(
                  initialValue: _studentId,
                  decoration: const InputDecoration(labelText: 'Linked student'),
                  items: _children.map((child) => DropdownMenuItem(value: int.tryParse('${child['id']}'), child: Text(child['name']?.toString() ?? 'Student'))).toList(),
                  onChanged: (value) { setState(() => _studentId = value); _loadLocation(); },
                ),
                const SizedBox(height: 16),
                Card(child: Padding(padding: const EdgeInsets.all(16), child: Row(children: [
                  CircleAvatar(backgroundColor: hasLocation ? AppTheme.teal : Colors.grey, child: Icon(hasLocation ? Icons.location_on : Icons.location_off, color: Colors.white)),
                  const SizedBox(width: 12),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(childName, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 17)),
                    const SizedBox(height: 4),
                    Text(hasLocation ? (_location?['stale'] == true ? 'Last location is older than 5 minutes' : 'Location sharing is active') : 'Location sharing is not active'),
                  ])),
                ]))),
                const SizedBox(height: 16),
                if (hasLocation)
                  SizedBox(height: 300, child: ClipRRect(borderRadius: BorderRadius.circular(18), child: FlutterMap(
                    options: MapOptions(initialCenter: LatLng(latitude, longitude), initialZoom: 15),
                    children: [
                      TileLayer(urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', userAgentPackageName: 'staybuddy_student'),
                      MarkerLayer(markers: [Marker(point: LatLng(latitude, longitude), width: 56, height: 56, child: const Icon(Icons.location_pin, color: Colors.red, size: 50))]),
                    ],
                  )))
                else Container(height: 210, alignment: Alignment.center, decoration: BoxDecoration(color: Colors.grey.shade100, borderRadius: BorderRadius.circular(18)), child: Padding(padding: const EdgeInsets.all(22), child: Text(_error ?? 'Ask your child to enable location sharing from their Student Portal.', textAlign: TextAlign.center))),
                if (hasLocation) ...[
                  const SizedBox(height: 16),
                  Text('Coordinates: ${latitude.toStringAsFixed(5)}, ${longitude.toStringAsFixed(5)}'),
                  Text('Accuracy: ${_number(_location?['accuracy_m']).toStringAsFixed(0)} m'),
                  Text('Updated: ${_location?['reported_at'] ?? '—'}'),
                ],
                const SizedBox(height: 20),
                const Text('The student controls sharing and can stop it at any time. This screen refreshes every 30 seconds while open.', textAlign: TextAlign.center),
              ]),
            ),
    );
  }
}

extension _FirstOrNull<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
