import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  final Api _api = Api();
  List<dynamic> _notifications = const [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadNotifications();
  }

  Future<void> _loadNotifications() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final notifications = await _api.getNotifications();
      if (!mounted) return;
      setState(() {
        _notifications = notifications;
        _isLoading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = error.toString();
      });
    }
  }

  Future<void> _markRead(Map<String, dynamic> notification) async {
    if (notification['read_at'] != null) return;
    final id = int.tryParse(notification['id'].toString());
    if (id == null) return;

    try {
      final result = await _api.markNotificationRead(id);
      final updated = result['notification'] as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _notifications = _notifications
            .map((item) => item['id'] == id ? updated : item)
            .toList();
      });
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not update notification: $error')),
      );
    }
  }

  IconData _iconFor(String type) {
    switch (type) {
      case 'booking_status':
        return Icons.event_available_rounded;
      case 'complaint_status':
        return Icons.support_agent_rounded;
      case 'announcement':
        return Icons.campaign_rounded;
      default:
        return Icons.notifications_rounded;
    }
  }

  Color _colorFor(String type) {
    switch (type) {
      case 'booking_status':
        return AppTheme.teal;
      case 'complaint_status':
        return Colors.orange.shade700;
      case 'announcement':
        return const Color(0xFF7B3FC4);
      default:
        return AppTheme.textMuted;
    }
  }

  String _timestamp(dynamic value) {
    final timestamp = DateTime.tryParse(value?.toString() ?? '')?.toLocal();
    if (timestamp == null) return '';
    final now = DateTime.now();
    if (DateUtils.isSameDay(timestamp, now)) {
      final hour = timestamp.hour % 12 == 0 ? 12 : timestamp.hour % 12;
      final minute = timestamp.minute.toString().padLeft(2, '0');
      return '$hour:$minute ${timestamp.hour >= 12 ? 'PM' : 'AM'}';
    }
    return '${timestamp.day}/${timestamp.month}/${timestamp.year}';
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: AppBar(
          backgroundColor: AppTheme.tealDeep,
          foregroundColor: Colors.white,
          title: Text(
            'Notifications',
            style: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w700),
          ),
        ),
        body: RefreshIndicator(
          color: AppTheme.teal,
          onRefresh: _loadNotifications,
          child: _buildBody(),
        ),
      ),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const SizedBox(height: 120),
          Icon(
            Icons.error_outline_rounded,
            size: 42,
            color: Colors.red.shade400,
          ),
          const SizedBox(height: 12),
          Text(
            'Notifications could not be loaded.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          Text(
            _error!,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
          ),
          const SizedBox(height: 16),
          Center(
            child: FilledButton.icon(
              onPressed: _loadNotifications,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Retry'),
            ),
          ),
        ],
      );
    }
    if (_notifications.isEmpty) {
      return ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const SizedBox(height: 120),
          Icon(
            Icons.notifications_none_rounded,
            size: 52,
            color: AppTheme.textMuted,
          ),
          const SizedBox(height: 12),
          Text(
            'You are all caught up',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(
            'Booking, complaint, and hostel updates will appear here.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
          ),
        ],
      );
    }

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      itemCount: _notifications.length,
      separatorBuilder: (context, index) => const SizedBox(height: 8),
      itemBuilder: (context, index) {
        final notification = _notifications[index] as Map<String, dynamic>;
        final type = notification['type']?.toString() ?? '';
        final unread = notification['read_at'] == null;
        final color = _colorFor(type);
        return Material(
          color: unread ? color.withValues(alpha: 0.08) : AppTheme.card,
          borderRadius: BorderRadius.circular(8),
          child: InkWell(
            borderRadius: BorderRadius.circular(8),
            onTap: () => _markRead(notification),
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.14),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(_iconFor(type), color: color, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          notification['title']?.toString() ?? 'Update',
                          style: GoogleFonts.dmSans(
                            fontSize: 14,
                            fontWeight: unread
                                ? FontWeight.w700
                                : FontWeight.w600,
                            color: AppTheme.textPrimary,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          notification['body']?.toString() ?? '',
                          style: GoogleFonts.dmSans(
                            fontSize: 12,
                            color: AppTheme.textMuted,
                            height: 1.35,
                          ),
                        ),
                        if (_timestamp(
                          notification['created_at'],
                        ).isNotEmpty) ...[
                          const SizedBox(height: 7),
                          Text(
                            _timestamp(notification['created_at']),
                            style: GoogleFonts.dmSans(
                              fontSize: 11,
                              color: AppTheme.textMuted,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  if (unread)
                    Container(
                      width: 8,
                      height: 8,
                      margin: const EdgeInsets.only(left: 8, top: 4),
                      decoration: BoxDecoration(
                        color: color,
                        shape: BoxShape.circle,
                      ),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
