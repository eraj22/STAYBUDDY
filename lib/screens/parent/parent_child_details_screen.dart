import 'package:flutter/material.dart';
import '../../api.dart';

class ParentChildDetailsScreen extends StatefulWidget {
  const ParentChildDetailsScreen({super.key});
  @override
  State<ParentChildDetailsScreen> createState() => _ParentChildDetailsScreenState();
}

class _ParentChildDetailsScreenState extends State<ParentChildDetailsScreen> {
  bool _loading = true;
  String? _error;
  Map<String, dynamic> _overview = const {};
  @override
  void initState() { super.initState(); _load(); }
  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final children = await Api().getParentChildren();
      if (children.isEmpty) throw Exception('No linked student was found.');
      final child = Map<String, dynamic>.from(children.first as Map);
      final id = int.tryParse(child['id']?.toString() ?? '');
      if (id == null) throw Exception('The linked student record is invalid.');
      final overview = await Api().getParentChildOverview(id);
      if (mounted) setState(() => _overview = overview);
    } catch (error) { if (mounted) setState(() => _error = error.toString().replaceFirst('Exception: ', '')); }
    finally { if (mounted) setState(() => _loading = false); }
  }
  String _value(dynamic value, [String fallback = 'Not provided']) => value == null || value.toString().isEmpty ? fallback : value.toString();
  @override
  Widget build(BuildContext context) {
    final student = _overview['student'] is Map ? Map<String, dynamic>.from(_overview['student'] as Map) : const <String, dynamic>{};
    final bookings = List<dynamic>.from(_overview['bookings'] as List? ?? const []);
    final booking = bookings.isEmpty ? const <String, dynamic>{} : Map<String, dynamic>.from(bookings.first as Map);
    return Scaffold(appBar: AppBar(title: const Text('Child Details')), body: RefreshIndicator(onRefresh: _load, child: _loading ? const Center(child: CircularProgressIndicator()) : _error != null ? ListView(children: [Padding(padding: const EdgeInsets.all(24), child: Column(children: [Text(_error!), FilledButton(onPressed: _load, child: const Text('Retry'))]))]) : ListView(padding: const EdgeInsets.all(16), children: [
      Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(_value(student['name'], 'Linked student'), style: Theme.of(context).textTheme.headlineSmall), const SizedBox(height: 12), Text('Email: ${_value(student['email'])}'), Text('Phone: ${_value(student['phone'])}'), Text('University: ${_value(student['university'])}'), Text('Department: ${_value(student['department'])}')]))),
      const SizedBox(height: 12), Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Current Hostel', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)), const SizedBox(height: 12), Text('Hostel: ${_value(booking['hostel_name'])}'), Text('Address: ${_value(booking['hostel_address'])}'), Text('Room: ${_value(booking['room_number'])}'), Text('Room type: ${_value(booking['room_type'])}'), Text('Booking status: ${_value(booking['status'])}'), Text('Checked in: ${_value(booking['checked_in_at'])}')]))),
    ])));
  }
}
